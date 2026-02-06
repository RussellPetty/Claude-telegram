#!/usr/bin/env python3
"""Telegram bot that bridges messages to Claude Code CLI with streaming updates."""

import asyncio
import json
import logging
import os
import uuid

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
WORKING_DIR = os.environ.get("CLAUDE_WORKING_DIR", os.path.expanduser("~"))
TIMEOUT = 300  # 5 minutes
MAX_MSG_LEN = 4096  # Telegram message limit

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

continue_session = True
active_proc = None  # Currently running Claude subprocess
sent_message_ids: list[int] = []  # Track bot-sent messages for clearing


async def send_chunks(chat, text: str) -> None:
    """Send text, splitting into chunks if needed."""
    if not text:
        return
    for i in range(0, len(text), MAX_MSG_LEN):
        msg = await chat.send_message(text[i : i + MAX_MSG_LEN], parse_mode=None)
        sent_message_ids.append(msg.message_id)


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


async def run_claude_streaming(prompt: str, chat, reply_to) -> None:
    """Run claude with stream-json output, sending updates as separate messages."""
    global continue_session, active_proc

    cmd = [
        "claude", "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if continue_session:
        cmd.append("--continue")

    # After this invocation, always continue
    continue_session = True

    working_msg = await reply_to.reply_text("⏳ Working on it...")
    sent_message_ids.append(working_msg.message_id)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=WORKING_DIR,
            limit=10 * 1024 * 1024,  # 10 MB line limit for large JSON output
        )
        active_proc = proc
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
                    if etype == "assistant":
                        msg = event.get("message", {})
                        content = msg.get("content", [])
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            # Only send tool use summaries during processing
                            if block.get("type") == "tool_use":
                                name = block.get("name", "")
                                # Only notify for Bash commands
                                if name == "Bash":
                                    summary = format_tool_use(block)
                                    if summary:
                                        await send_chunks(chat, summary)

                    elif etype == "result":
                        # Final response text — this is the only text we send
                        text = event.get("result", "").strip()
                        if text:
                            await send_chunks(chat, text)
                except Exception as e:
                    logger.warning("Error processing event: %s", e)
                    continue

        # Wait for process to finish
        await proc.wait()

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if stderr_text:
                await send_chunks(chat, f"⚠️ Claude exited with errors:\n{stderr_text[:3000]}")

    except asyncio.TimeoutError:
        await chat.send_message("⏰ Claude timed out after 5 minutes.")
        proc.kill()
    except Exception as e:
        await chat.send_message(f"❌ Error: {e}")
    finally:
        active_proc = None
        typing_active = False
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start."""
    msg = await update.message.reply_text(
        "Hello! I'm a bridge to Claude Code. Send me a message and I'll forward it "
        "to Claude. Use /new to start a fresh session, or /stop to stop what's running."
    )
    sent_message_ids.append(msg.message_id)


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new — reset session and clear bot messages from chat."""
    global continue_session, sent_message_ids
    continue_session = False

    chat_id = update.message.chat_id
    deleted = 0
    for msg_id in sent_message_ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            deleted += 1
        except Exception:
            pass  # Message may already be deleted or too old (>48h)
    sent_message_ids = []

    await update.message.reply_text("Session cleared. Next message starts fresh.")


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop — kill the running Claude process without resetting the session."""
    global active_proc
    if active_proc is None:
        msg = await update.message.reply_text("Nothing is running right now.")
        sent_message_ids.append(msg.message_id)
        return
    try:
        active_proc.kill()
    except ProcessLookupError:
        pass
    msg = await update.message.reply_text("🛑 Stopped.")
    sent_message_ids.append(msg.message_id)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages."""
    text = update.message.text
    if not text:
        return
    await run_claude_streaming(text, update.message.chat, update.message)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — save image, ask Claude to read it."""
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

    await run_claude_streaming(prompt, update.message.chat, update.message)


def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("new", new_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
