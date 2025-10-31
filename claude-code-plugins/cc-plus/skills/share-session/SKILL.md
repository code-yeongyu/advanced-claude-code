---
name: share-session
description: Convert and share Claude Code conversation sessions as readable markdown files. Use when user wants to share a session transcript, export conversation history, or create a shareable markdown document from a Claude Code session. Triggered by requests like "share this session", "export conversation", "convert session to markdown".
---

# Share Session

## Overview

Convert Claude Code sessions into readable markdown format for easy sharing. This skill finds sessions by fuzzy matching todo items and generates well-formatted markdown documents.
If this is loaded by user's explicit request but no comments there, just execute followings.

## Workflow

### Step 1: Create Session Identifier Todo

Use TodoWrite to add a todo item with an identifiable value from the session:

```
Use TodoWrite tool to add:
"get session id of {anything identifiable}"
```

Examples of identifiable values:
- Recent user message: "get session id of create python tool"
- Task description: "get session id of pdf converter"
- Timestamp: "get session id of today's morning work"

### Step 2: Run share_session.py

Run the single unified script (always keep in mind of where the script is located):

```bash
uv run --script scripts/share_session.py "your identifiable query"
```

The script automatically does everything:
- Searches todos using fuzzy matching (60% threshold)
- Locates transcript at `~/.claude/projects/*/{session-id}.jsonl`
- Merges pre-compact backups if they exist
- Fetches latest pricing data from LiteLLM
- Converts to markdown with full statistics
- Saves to `/tmp/claude-code-sessions/{session-id}-{timestamp}.md`
- Copies the file path to clipboard
- Displays success message with cost breakdown

### Step 3: Output

The script displays:
```
✅ Markdown saved to:
/tmp/claude-code-sessions/{session-id}-{timestamp}.md

💰 Total Session Cost: $X.XXXXXX

📋 The path has been copied to your clipboard.
```

## Generated Markdown Format

The single script generates comprehensive markdown with:

- 📊 Session metadata (ID, timestamp, message count)
- 💬 User messages with timestamps
- 🤖 Assistant responses
- 🧠 Thinking process (when available)
- 🔧 Tool usage details
- 📈 Token usage per message
- 💰 Total session cost breakdown
- 📊 Token usage statistics (input, output, cache creation, cache read)
- 🎯 Cache hit rate percentage
- 📉 Average cost per message
- ⏱️ Session timeline (total time, LLM active time, wait time, utilization)
- 🔄 Multi-model support (different models per message)

## Script

### share_session.py

**The only script you need.** Does everything from search to markdown generation.

**Usage:**
```bash
uv run --script scripts/share_session.py <query>
```

**Complete features:**
- Fuzzy search through todo files (60% threshold)
- Automatic pre-compact backup merging
- Real-time pricing from LiteLLM (1,679+ models)
- Accurate cache token cost calculation
- Session timeline tracking
- Multi-model session support
- Clipboard integration (macOS)
- Rich terminal output with progress indicators
- TypedDict-based type safety

**Output:** File path (stdout) + clipboard

**Exit codes:**
- 0: Success
- 1: Session not found or conversion failed

## Error Handling

**No session found:**
- Check todo item was added correctly
- Verify query matches todo content
- Try more specific identifiable value

**Transcript not found:**
- Confirm session ID is correct
- Check `~/.claude/projects/` directory exists
- Verify transcript file exists

**Conversion failed:**
- Check transcript file is valid JSONL
- Review error message from stderr
- Check internet connection (pricing data fetch)

**Clipboard copy failed:**
- Warning displayed but conversion continues
- File path still available in stdout
- Manual copy may be needed
