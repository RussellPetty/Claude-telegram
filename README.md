# Telegram Claude Bot

A Telegram bot that bridges messages to Claude Code CLI with streaming updates.

## Setup

1. Clone the repo:
   ```bash
   git clone git@github.com:YOUR_USERNAME/telegram-claude-bot.git
   cd telegram-claude-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and add your bot token and working directory.

5. Run the bot:
   ```bash
   ./run-forever.sh
   ```

## Commands

- `/start` — Welcome message
- `/new` — Start a fresh Claude session
- `/stop` — Stop the currently running Claude process
- Send any text or photo to forward it to Claude
