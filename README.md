# CC Plus

Essential productivity features that should exist in Claude Code but don't - packaged as convenient plugins.

## Features

### 🔄 Session Backup Hook
Automatically backs up your Claude Code transcripts before compaction. Never lose your conversation history!

- **Hook**: `pre_compact.py` - Triggers before session compaction
- **Backup Location**: `~/.claude/sessions/`

### 📤 Share Session Skill
Export and share Claude Code sessions as readable markdown files.

- **Command**: Use the `share-session` skill
- **Output**: Clean markdown format perfect for documentation or sharing

### 🔀 PR Creator Skill
Create GitHub Pull Requests directly from Claude Code with proper templates.

- **Command**: Use the `pr-creator` skill
- **Features**: Auto-detects PR templates, formats commit history

## Installation

### Option 1: Install via Marketplace (Recommended)

```bash
# Clone the repository
git clone https://github.com/code-yeongyu/cc-plus.git

# Install the marketplace
claude plugin install cc-plus/.claude-plugin/marketplace.json
```

### Option 2: Install Plugin Directly

```bash
# Clone the repository
git clone https://github.com/code-yeongyu/cc-plus.git

# Copy plugin to your Claude Code plugins directory
cp -r cc-plus/claude-code-plugins/cc-plus ~/.claude/plugins/
```

### Option 3: Manual Installation (For Development)

```bash
# Clone the repository
git clone https://github.com/code-yeongyu/cc-plus.git
cd cc-plus

# Symlink individual components
mkdir -p ~/.claude/hooks ~/.claude/skills

# Hooks
ln -s "$(pwd)/claude-code-plugins/cc-plus/hooks/pre_compact.py" ~/.claude/hooks/
ln -s "$(pwd)/claude-code-plugins/cc-plus/hooks/pre-compact" ~/.claude/hooks/

# Skills
ln -s "$(pwd)/claude-code-plugins/cc-plus/skills/share-session" ~/.claude/skills/
ln -s "$(pwd)/claude-code-plugins/cc-plus/skills/pr-creator" ~/.claude/skills/
```

## Usage

### Session Backup
No setup needed! Your sessions will be automatically backed up before compaction.

```bash
# Check your backups
ls ~/.claude/sessions/
```

### Share Session
```bash
# In Claude Code chat
/share-session

# Or activate the skill
share-session
```

### Create PR
```bash
# In Claude Code chat
pr-creator

# Follow the interactive prompts
```

## CI/CD Integration

Use these plugins in GitHub Actions:

```yaml
- name: Setup Claude Code configuration
  run: |
    mkdir -p ~/.claude

    # Install from repository
    git clone https://github.com/code-yeongyu/cc-plus.git /tmp/cc-plus
    cp -r /tmp/cc-plus/claude-code-plugins/cc-plus/* ~/.claude/
```


## File Structure

```
cc-plus/
├── .claude-plugin/
│   └── marketplace.json                 # Marketplace definition
├── claude-code-plugins/
│   └── cc-plus/
│       ├── plugin.json                  # Plugin metadata
│       ├── hooks/
│       │   ├── pre_compact.py          # Hook orchestrator
│       │   └── pre-compact/
│       │       └── backup_transcript.py # Backup implementation
│       └── skills/
│           ├── share-session/
│           │   ├── SKILL.md
│           │   └── scripts/
│           └── pr-creator/
│               ├── SKILL.md
│               └── scripts/
└── README.md
```

## License

MIT

## Author

**YeonGyu Kim**
- GitHub: [@code-yeongyu](https://github.com/code-yeongyu)
- Email: public.kim.yeon.gyu@gmail.com

## Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest new essential features
- Submit PRs
