---
name: share
description: Export current Claude Code session as readable markdown file. Quick shortcut for share-session skill.
---

# Share Session Command

Export the current Claude Code conversation session to a shareable markdown file.

## Instructions

Execute the share-session skill to convert this session into markdown format:

1. Use the Skill tool to invoke the share-session skill
2. The skill will automatically:
   - Create a session identifier todo with current context
   - Search for the session using fuzzy matching
   - Convert the transcript to markdown with full statistics
   - Save to `/tmp/claude-code-sessions/` with timestamp
   - Copy the file path to clipboard
   - Display cost breakdown and session metrics

## Output

You will receive:
- 📄 Markdown file path (copied to clipboard)
- 💰 Total session cost breakdown
- 📊 Token usage statistics
- ⏱️ Session timeline metrics
- 🎯 Cache hit rate percentage

Simply invoke the skill - no additional parameters needed.
