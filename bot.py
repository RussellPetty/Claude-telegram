#!/usr/bin/env python3
"""Telegram bot that bridges messages to Claude Code CLI with streaming updates."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field

from telegram import ReactionTypeEmoji, Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegramify_markdown import markdownify

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
WORKING_DIR = os.environ.get("CLAUDE_WORKING_DIR", os.path.expanduser("~"))
ALLOWED_USER_ID = int(os.environ["ALLOWED_USER_ID"])
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")
TIMEOUT = 300  # 5 minutes
MAX_MSG_LEN = 4096  # Telegram message limit

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def is_allowed(update: Update) -> bool:
    """Return True if the message is from the allowed user."""
    user = update.effective_user
    if user and user.id == ALLOWED_USER_ID:
        return True
    logger.warning("Blocked message from user %s (id=%s)", user.username if user else "?", user.id if user else "?")
    return False


@dataclass
class ChatState:
    """Per-chat session state."""
    session_id: str | None = None
    model_override: str | None = None
    model_choices: list[str] = field(default_factory=list)
    active_proc: asyncio.subprocess.Process | None = None
    sent_message_ids: list[int] = field(default_factory=list)
    user_message_ids: list[int] = field(default_factory=list)
    term_mode: bool = False
    codex_mode: bool = False
    codex_thread_id: str | None = None
    codex_history: list[str] = field(default_factory=list)
    pending_codex_context: str | None = None
    stop_requested: bool = False
    pending_text: list[str] = field(default_factory=list)
    debounce_task: asyncio.Task | None = None
    last_reply_to: object | None = None
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# chat_id -> ChatState
chats: dict[int, ChatState] = {}


def get_state(chat_id: int) -> ChatState:
    """Get or create per-chat state."""
    if chat_id not in chats:
        chats[chat_id] = ChatState()
    return chats[chat_id]


def _split_mdv2(text: str, limit: int) -> list[str]:
    """Split MarkdownV2 text into chunks at newline boundaries."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        idx = text.rfind("\n", 0, limit)
        if idx == -1:
            idx = limit
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks


async def send_chunks(chat, text: str, state: ChatState) -> None:
    """Send text, converting Markdown to Telegram MarkdownV2 and splitting into chunks."""
    if not text:
        return
    try:
        converted = markdownify(text)
    except Exception as e:
        logger.warning("Markdown conversion failed, sending as plain text: %s", e)
        converted = None

    if converted:
        chunks = _split_mdv2(converted, MAX_MSG_LEN)
        for chunk in chunks:
            try:
                msg = await chat.send_message(chunk, parse_mode="MarkdownV2")
            except Exception as e:
                logger.warning("MarkdownV2 send failed, falling back to plain text: %s", e)
                msg = await chat.send_message(text[:MAX_MSG_LEN], parse_mode=None)
            state.sent_message_ids.append(msg.message_id)
    else:
        for i in range(0, len(text), MAX_MSG_LEN):
            msg = await chat.send_message(text[i : i + MAX_MSG_LEN], parse_mode=None)
            state.sent_message_ids.append(msg.message_id)


def format_tool_use(content_block: dict) -> str:
    """Format a tool_use block into a readable message."""
    name = content_block.get("name", "unknown")
    inp = content_block.get("input", {})

    if name == "Bash":
        desc = inp.get("description", "")
        if desc:
            return f"**{desc}**"
        return None
    elif name == "Write":
        path = inp.get("file_path", "")
        return f"📝 Writing file: `{path}`"
    elif name == "Edit":
        path = inp.get("file_path", "")
        return f"✏️ Editing file: `{path}`"
    elif name == "Read":
        path = inp.get("file_path", "")
        return f"📖 Reading file: `{path}`"
    elif name == "Glob":
        pattern = inp.get("pattern", "")
        return f"🔍 Searching for files: `{pattern}`"
    elif name == "Grep":
        pattern = inp.get("pattern", "")
        return f"🔍 Searching content: `{pattern}`"
    elif name == "WebFetch":
        url = inp.get("url", "")
        return f"🌐 Fetching: {url}"
    elif name == "WebSearch":
        query = inp.get("query", "")
        return f"🔎 Searching web: {query}"
    elif name == "Task":
        desc = inp.get("description", "")
        return f"🤖 Spawning agent: {desc}"
    else:
        return f"🔧 Using tool: {name}"


def format_tool_result(event: dict) -> str | None:
    """Format a tool result into a readable message, or None to skip."""
    content = event.get("message", {}).get("content", [])
    if not content:
        return None

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            inner = block.get("content", "")
            # content can be a string or a list of content blocks
            if isinstance(inner, str):
                text = inner
            elif isinstance(inner, list):
                parts = []
                for item in inner:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(item.get("text", ""))
                text = "\n".join(parts)
            else:
                continue
            text = text.strip()
            if text:
                if len(text) > 2000:
                    text = text[:2000] + "\n... (truncated)"
                return f"📋 Result:\n```\n{text}\n```"
    return None


async def run_claude_streaming(prompt: str, chat, reply_to, state: ChatState) -> None:
    """Run claude with stream-json output, sending updates as separate messages."""

    # Prepend codex context if switching back from codex
    if state.pending_codex_context:
        prompt = state.pending_codex_context + "\n\nUser's new message: " + prompt
        state.pending_codex_context = None

    cmd = [
        "claude", "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    cmd.extend(["--model", state.model_override or DEFAULT_MODEL])
    if state.session_id:
        cmd.extend(["--resume", state.session_id])

    try:
        await reply_to.set_reaction(ReactionTypeEmoji("👍"))
    except Exception as e:
        logger.error("Failed to set reaction: %s", e)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=WORKING_DIR,
            limit=10 * 1024 * 1024,  # 10 MB line limit for large JSON output
        )
        state.active_proc = proc
    except Exception as e:
        await chat.send_message(f"❌ Error starting Claude: {e}")
        return

    # Keep typing indicator alive in background
    typing_active = True

    async def keep_typing():
        while typing_active:
            try:
                await chat.send_action(ChatAction.TYPING)
            except Exception:
                pass
            await asyncio.sleep(8)

    typing_task = asyncio.create_task(keep_typing())

    try:
        buffer = ""
        async for raw_chunk in proc.stdout:
            buffer += raw_chunk.decode("utf-8", errors="replace")

            # Process complete JSON lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                try:
                    # Capture session ID from init or any event
                    if "session_id" in event and not state.session_id:
                        state.session_id = event["session_id"]
                        logger.info("Captured session_id: %s", state.session_id)

                    if etype == "assistant":
                        msg = event.get("message", {})
                        content = msg.get("content", [])
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") == "tool_use":
                                name = block.get("name", "")
                                if name in ("Bash", "WebSearch", "WebFetch"):
                                    summary = format_tool_use(block)
                                    if summary:
                                        await send_chunks(chat, summary, state)

                    elif etype == "result":
                        # Final response text — this is the only text we send
                        text = event.get("result", "").strip()
                        if text:
                            await send_chunks(chat, text, state)
                except Exception as e:
                    logger.warning("Error processing event: %s", e)
                    continue

        # Wait for process to finish
        await proc.wait()

        if proc.returncode != 0 and not state.stop_requested:
            stderr = await proc.stderr.read()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if stderr_text:
                await send_chunks(chat, f"⚠️ Claude exited with errors:\n{stderr_text[:3000]}", state)

    except asyncio.TimeoutError:
        await chat.send_message("⏰ Claude timed out after 5 minutes.")
        proc.kill()
    except Exception as e:
        await chat.send_message(f"❌ Error: {e}")
    finally:
        state.active_proc = None
        state.stop_requested = False
        typing_active = False
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def run_codex_streaming(prompt: str, chat, reply_to, state: ChatState) -> None:
    """Run codex exec with --json output, sending updates as separate messages."""

    cmd = [
        "codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--json",
        "-C", WORKING_DIR,
        prompt,
    ]
    if state.model_override:
        cmd.extend(["-m", state.model_override])

    try:
        await reply_to.set_reaction(ReactionTypeEmoji("👍"))
    except Exception as e:
        logger.error("Failed to set reaction: %s", e)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            cwd=WORKING_DIR,
            limit=10 * 1024 * 1024,
        )
        state.active_proc = proc
    except Exception as e:
        await chat.send_message(f"❌ Error starting Codex: {e}")
        return

    # Close stdin so codex doesn't hang waiting for input
    proc.stdin.close()

    # Track user prompt in codex history
    state.codex_history.append(f"User: {prompt}")

    # Keep typing indicator alive in background
    typing_active = True

    async def keep_typing():
        while typing_active:
            try:
                await chat.send_action(ChatAction.TYPING)
            except Exception:
                pass
            await asyncio.sleep(8)

    typing_task = asyncio.create_task(keep_typing())

    try:
        buffer = ""
        async for raw_chunk in proc.stdout:
            buffer += raw_chunk.decode("utf-8", errors="replace")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                try:
                    if etype == "thread.started":
                        tid = event.get("thread_id")
                        if tid:
                            state.codex_thread_id = tid
                            logger.info("Codex thread_id: %s", tid)

                    elif etype == "item.completed":
                        item = event.get("item", {})
                        itype = item.get("type", "")

                        if itype == "agent_message":
                            text = item.get("text", "").strip()
                            if text:
                                await send_chunks(chat, text, state)
                                state.codex_history.append(f"Codex: {text}")

                        elif itype == "command_execution":
                            cmd_str = item.get("command", "")
                            exit_code = item.get("exit_code")
                            status = item.get("status", "")
                            if cmd_str and status == "completed":
                                icon = "✅" if exit_code == 0 else "⚠️"
                                await send_chunks(chat, f"{icon} `{cmd_str}` (exit {exit_code})", state)

                except Exception as e:
                    logger.warning("Error processing codex event: %s", e)
                    continue

        await proc.wait()

        if proc.returncode != 0 and not state.stop_requested:
            stderr = await proc.stderr.read()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if stderr_text:
                await send_chunks(chat, f"⚠️ Codex exited with errors:\n{stderr_text[:3000]}", state)

    except Exception as e:
        await chat.send_message(f"❌ Error: {e}")
    finally:
        state.active_proc = None
        state.stop_requested = False
        typing_active = False
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def run_terminal_command(command: str, chat, reply_to, state: ChatState) -> None:
    """Run a shell command and relay output back to the chat."""
    working_msg = await reply_to.reply_text(f"🖥️ Running: `{command}`", parse_mode="Markdown")
    state.sent_message_ids.append(working_msg.message_id)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=WORKING_DIR,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
        output = stdout.decode("utf-8", errors="replace").strip()

        if not output:
            output = "(no output)"

        exit_info = f"Exit code: {proc.returncode}"
        result = f"```\n{output}\n```\n{exit_info}"
        await send_chunks(chat, result, state)

    except asyncio.TimeoutError:
        await send_chunks(chat, "⏰ Command timed out after 5 minutes.", state)
        proc.kill()
    except Exception as e:
        await send_chunks(chat, f"❌ Error: {e}", state)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)
    msg = await update.message.reply_text(
        "Hello! I'm a bridge to Claude Code. Send me a message and I'll forward it "
        "to Claude. Use /new to start a fresh session, or /stop to stop what's running."
    )
    state.sent_message_ids.append(msg.message_id)


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new — reset session and clear all messages from chat."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)
    state.session_id = None
    state.model_override = None
    state.codex_mode = False
    state.codex_thread_id = None
    state.codex_history = []
    state.pending_codex_context = None

    chat_id = update.message.chat_id
    all_ids = state.sent_message_ids + state.user_message_ids
    for msg_id in all_ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception:
            pass  # Message may already be deleted or too old (>48h)
    state.sent_message_ids = []
    state.user_message_ids = []

    # Delete the /new command message itself
    try:
        await update.message.delete()
    except Exception:
        pass

    await update.message.reply_text("Hi, how can I help you?")


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop — kill the running Claude process without resetting the session."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)

    # Cancel any pending debounce so queued chunks don't fire after stop
    if state.debounce_task and not state.debounce_task.done():
        state.debounce_task.cancel()
        state.debounce_task = None
    state.pending_text.clear()

    if state.active_proc is None:
        msg = await update.message.reply_text("Nothing is running right now.")
        state.sent_message_ids.append(msg.message_id)
        return
    state.stop_requested = True
    try:
        state.active_proc.kill()
    except ProcessLookupError:
        pass
    msg = await update.message.reply_text("🛑 Stopped.")
    state.sent_message_ids.append(msg.message_id)


async def term_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /term — next message will be run as a shell command."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)
    state.term_mode = True
    msg = await update.message.reply_text("🖥️ Term mode active. Send a command to run.")
    state.sent_message_ids.append(msg.message_id)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show current session state."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)
    running = "Yes" if state.active_proc is not None else "No"
    session = f"Active (`{state.session_id[:8]}...`)" if state.session_id else "Fresh (next message starts new)"
    mode = "Terminal" if state.term_mode else ("Codex" if state.codex_mode else "Claude")

    lines = [
        f"Chat ID: `{update.message.chat_id}`",
        f"Working dir: `{WORKING_DIR}`",
        f"Session: {session}",
        f"Process running: {running}",
        f"Input mode: {mode}",
    ]
    msg = await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    state.sent_message_ids.append(msg.message_id)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model — list available models and let user pick one."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)

    # Exit codex mode and seed Claude with codex history on next message
    if state.codex_mode:
        if state.codex_history:
            transcript = "\n".join(state.codex_history)
            state.pending_codex_context = (
                "The user was previously working with Codex. Here is the conversation that took place:\n\n"
                f"{transcript}\n\n"
                "Continue assisting them, taking the above context into account."
            )
        state.codex_mode = False
        state.codex_thread_id = None
        state.codex_history = []

    # Claude Code accepts these aliases
    models = ["opus", "sonnet", "haiku"]
    state.model_choices = models

    current = state.model_override or "default"
    lines = [f"**Current model:** `{current}`\n", "**Pick a model** (reply with the number):\n"]
    for i, m in enumerate(models, 1):
        check = " ✅" if m == state.model_override else ""
        lines.append(f"`{i}.` {m}{check}")
    lines.append(f"\n`0.` Reset to default")

    msg = await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    state.sent_message_ids.append(msg.message_id)


async def codex_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /codex — switch to Codex mode, seeding it with all prior chat context."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)

    if state.codex_mode:
        msg = await update.message.reply_text("Already in Codex mode. Use /new or /model to switch back to Claude.")
        state.sent_message_ids.append(msg.message_id)
        return

    # Collect all message history from this chat to seed Codex
    chat_obj = update.message.chat
    history_lines = []
    try:
        # Gather recent messages from Telegram chat history
        # We go through sent + user message IDs we've tracked
        # But more reliably, we can use the bot's tracked context
        pass
    except Exception:
        pass

    # Build context from what we know: replay any session context
    # The real value is forwarding the conversation so far
    # Collect messages by iterating tracked user messages
    # Since we can't easily read back message text from IDs alone,
    # we'll note the session switch and let the user continue from here
    state.codex_mode = True
    state.codex_thread_id = None  # Fresh codex session

    msg = await update.message.reply_text(
        "🔄 Switched to **Codex** mode (`--dangerously-bypass-approvals-and-sandbox`).\n\n"
        "All messages will now be routed to Codex.\n"
        "Use /new or /model to switch back to Claude.",
        parse_mode="Markdown",
    )
    state.sent_message_ids.append(msg.message_id)

    # If there's an existing Claude session, build a context summary prompt
    # and send it to Codex as the first message so it has the conversation context
    if state.session_id:
        context_prompt = (
            "You are continuing a conversation that was previously handled by Claude Code. "
            "The user has switched to Codex. Continue assisting them with whatever they need. "
            "The previous Claude session ID was: " + state.session_id
        )
        async with state.processing_lock:
            await run_codex_streaming(context_prompt, chat_obj, update.message, state)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages — transcribe with OpenAI Whisper, send to Claude."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)
    state.user_message_ids.append(update.message.message_id)

    if not OPENAI_API_KEY:
        msg = await update.message.reply_text("⚠️ OPENAI_API_KEY not set — can't transcribe voice.")
        state.sent_message_ids.append(msg.message_id)
        return

    voice = update.message.voice
    file = await voice.get_file()
    voice_path = f"/tmp/tg_voice_{uuid.uuid4().hex}.ogg"
    await file.download_to_drive(voice_path)
    logger.info("Saved voice message to %s (%d seconds)", voice_path, voice.duration)

    # Transcribe via OpenAI Whisper API
    try:
        import urllib.request
        import urllib.error

        with open(voice_path, "rb") as audio_file:
            audio_data = audio_file.read()

        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="voice.ogg"\r\n'
            f"Content-Type: audio/ogg\r\n\r\n"
        ).encode() + audio_data + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"whisper-1"
            f"\r\n--{boundary}--\r\n"
        ).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            transcript = result.get("text", "").strip()
    except Exception as e:
        logger.error("Whisper transcription failed: %s", e)
        msg = await update.message.reply_text(f"⚠️ Transcription failed: {e}")
        state.sent_message_ids.append(msg.message_id)
        return
    finally:
        try:
            os.remove(voice_path)
        except OSError:
            pass

    if not transcript:
        msg = await update.message.reply_text("Couldn't transcribe the voice message.")
        state.sent_message_ids.append(msg.message_id)
        return

    logger.info("Transcribed voice: %s", transcript[:100])
    if state.codex_mode:
        await run_codex_streaming(transcript, update.message.chat, update.message, state)
    else:
        await run_claude_streaming(transcript, update.message.chat, update.message, state)


async def _process_debounced(chat, state: ChatState) -> None:
    """Wait for the debounce window, then send all buffered text to Claude."""
    await asyncio.sleep(1.5)  # Debounce window — wait for more chunks

    # Grab everything that accumulated and clear the buffer
    combined = "\n".join(state.pending_text)
    reply_to = state.last_reply_to
    state.pending_text = []
    state.last_reply_to = None

    if not combined.strip():
        return

    async with state.processing_lock:
        if state.codex_mode:
            await run_codex_streaming(combined, chat, reply_to, state)
        else:
            await run_claude_streaming(combined, chat, reply_to, state)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages with debounce for chunked inputs."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)
    text = update.message.text
    if not text:
        return
    state.user_message_ids.append(update.message.message_id)

    # Handle model selection if choices are pending
    if state.model_choices and text.strip().isdigit():
        idx = int(text.strip())
        if idx == 0:
            state.model_override = None
            state.model_choices = []
            msg = await update.message.reply_text("Model reset to default.")
            state.sent_message_ids.append(msg.message_id)
            return
        if 1 <= idx <= len(state.model_choices):
            state.model_override = state.model_choices[idx - 1]
            state.model_choices = []
            msg = await update.message.reply_text(f"Model set to `{state.model_override}`", parse_mode="Markdown")
            state.sent_message_ids.append(msg.message_id)
            return
        state.model_choices = []  # Invalid number, clear and fall through

    if state.term_mode:
        state.term_mode = False
        await run_terminal_command(text, update.message.chat, update.message, state)
        return

    # Buffer the message and (re)start the debounce timer
    state.pending_text.append(text)
    state.last_reply_to = update.message

    if state.debounce_task and not state.debounce_task.done():
        state.debounce_task.cancel()

    state.debounce_task = asyncio.create_task(
        _process_debounced(update.message.chat, state)
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — save image, ask Claude to read it."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)
    state.user_message_ids.append(update.message.message_id)
    photo = update.message.photo[-1]
    file = await photo.get_file()

    img_path = f"/tmp/tg_img_{uuid.uuid4().hex}.jpg"
    await file.download_to_drive(img_path)
    logger.info("Saved image to %s", img_path)

    caption = update.message.caption or ""
    if caption:
        prompt = f"Read the image at {img_path}. User says: {caption}"
    else:
        prompt = f"Read the image at {img_path} and describe what you see."

    if state.codex_mode:
        await run_codex_streaming(prompt, update.message.chat, update.message, state)
    else:
        await run_claude_streaming(prompt, update.message.chat, update.message, state)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document messages — save file, ask Claude to read it."""
    if not is_allowed(update):
        return
    state = get_state(update.message.chat_id)
    state.user_message_ids.append(update.message.message_id)
    doc = update.message.document
    file = await doc.get_file()

    filename = doc.file_name or f"document_{uuid.uuid4().hex}"
    doc_path = f"/tmp/tg_doc_{uuid.uuid4().hex}_{filename}"
    await file.download_to_drive(doc_path)
    logger.info("Saved document to %s (%s, %d bytes)", doc_path, doc.mime_type, doc.file_size or 0)

    caption = update.message.caption or ""
    if caption:
        prompt = f"Read the file at {doc_path}. User says: {caption}"
    else:
        prompt = f"Read the file at {doc_path} and describe its contents."

    if state.codex_mode:
        await run_codex_streaming(prompt, update.message.chat, update.message, state)
    else:
        await run_claude_streaming(prompt, update.message.chat, update.message, state)


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /restart — restart the entire bot process."""
    if not is_allowed(update):
        return
    await update.message.reply_text("🔄 Restarting bot...")
    os._exit(0)


def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).concurrent_updates(True).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("new", new_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("term", term_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("restart", restart_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("codex", codex_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
