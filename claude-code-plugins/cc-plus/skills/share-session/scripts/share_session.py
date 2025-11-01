#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "~=3.12"
# dependencies = [
#     "orjson",
#     "thefuzz",
#     "rich",
# ]
# ///

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import orjson
from rich.console import Console
from thefuzz import fuzz

console = Console()


class SessionUsageData(TypedDict):
    sessionId: str
    inputTokens: int
    outputTokens: int
    cacheCreationTokens: int
    cacheReadTokens: int
    totalTokens: int
    totalCost: float
    lastActivity: str
    modelsUsed: list[str]
    modelBreakdowns: NotRequired[list[dict[str, Any]]]


def extract_session_id_from_filename(filename: str) -> str | None:
    if not filename.endswith(".json"):
        return None
    parts = filename.replace(".json", "").split("-agent-")
    if len(parts) >= 1:
        return parts[0]
    return None


def search_todos(query: str, todos_dir: Path) -> str | None:
    if not todos_dir.exists():
        return None

    best_match_score = 0
    best_session_id = None

    for todo_file in todos_dir.glob("*.json"):
        try:
            todos = orjson.loads(todo_file.read_bytes())
        except (orjson.JSONDecodeError, OSError):
            continue

        for todo in todos:
            content = todo.get("content", "")
            score = fuzz.partial_ratio(query.lower(), content.lower())

            if score > best_match_score:
                best_match_score = score
                best_session_id = extract_session_id_from_filename(todo_file.name)

    if best_match_score >= 60:
        return best_session_id
    return None


def find_transcript_path(session_id: str) -> Path | None:
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return None

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        transcript_file = project_dir / f"{session_id}.jsonl"
        if transcript_file.exists():
            return transcript_file

    return None


def find_pre_compact_backups(session_id: str) -> list[Path]:
    backup_dir = Path.home() / ".claude" / "pre-compact-session-histories"
    if not backup_dir.exists():
        return []

    backups = list(backup_dir.glob(f"{session_id}-*.jsonl"))
    backups.sort(key=lambda p: p.name)
    return backups


def create_merged_transcript(session_id: str, current_transcript: Path) -> Path | None:
    backups = find_pre_compact_backups(session_id)

    if not backups:
        return current_transcript

    merged_dir = Path("/tmp/claude-code-merged-transcripts")
    merged_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    merged_file = merged_dir / f"{session_id}-merged-{timestamp_str}.jsonl"

    try:
        with merged_file.open("w", encoding="utf-8") as outfile:
            for i, backup_path in enumerate(backups):
                if i > 0:
                    compact_marker = orjson.dumps({"type": "compact_marker"}).decode()
                    outfile.write(compact_marker + "\n")

                with backup_path.open("r", encoding="utf-8") as infile:
                    outfile.write(infile.read())

            compact_marker = orjson.dumps({"type": "compact_marker"}).decode()
            outfile.write(compact_marker + "\n")

            with current_transcript.open("r", encoding="utf-8") as infile:
                outfile.write(infile.read())

        return merged_file
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to merge transcripts: {e}[/yellow]")
        return current_transcript


def fetch_session_usage_from_ccusage(session_id: str) -> SessionUsageData | None:
    try:
        result = subprocess.run(
            ["bunx", "--bun", "ccusage", "session", "-i", session_id, "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

        data = orjson.loads(result.stdout)

        if not isinstance(data, dict):
            console.print("[red]Error: Unexpected ccusage output format[/red]")
            return None

        total_input = 0
        total_output = 0
        total_cache_creation = 0
        total_cache_read = 0
        models_used_set: set[str] = set()

        entries = data.get("entries", [])
        for entry in entries:
            total_input += entry.get("inputTokens", 0)
            total_output += entry.get("outputTokens", 0)
            total_cache_creation += entry.get("cacheCreationTokens", 0)
            total_cache_read += entry.get("cacheReadTokens", 0)

            model = entry.get("model")
            if model:
                models_used_set.add(model)

        total_tokens = data.get("totalTokens", 0)
        total_cost = data.get("totalCost", 0.0)

        last_activity = ""
        if entries:
            last_entry = entries[-1]
            last_timestamp = last_entry.get("timestamp", "")
            if last_timestamp:
                last_activity = last_timestamp.split("T")[0]

        return SessionUsageData(
            sessionId=data.get("sessionId", session_id),
            inputTokens=total_input,
            outputTokens=total_output,
            cacheCreationTokens=total_cache_creation,
            cacheReadTokens=total_cache_read,
            totalTokens=total_tokens,
            totalCost=total_cost,
            lastActivity=last_activity,
            modelsUsed=sorted(models_used_set),
        )

    except subprocess.TimeoutExpired:
        console.print("[red]Error: ccusage command timed out[/red]")
        return None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error calling ccusage: {e.stderr}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error fetching session usage: {e}[/red]")
        return None


def escape_xml_tags(text: str) -> str:
    return text.replace("<", r"\<").replace(">", r"\>")


def parse_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        local_dt = dt.astimezone()
        return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, AttributeError):
        return ts


def parse_timestamp_to_datetime(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if secs > 0:
            return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
        else:
            return f"{int(hours)}h {int(minutes)}m"


def format_tool_parameters(params: dict[str, Any]) -> str:
    if not params:
        return "_No parameters_"

    lines = []
    for key, value in params.items():
        match value:
            case str():
                if len(value) > 100:
                    lines.append(f"**{key}**: `{value[:100]}...`")
                else:
                    lines.append(f"**{key}**: `{value}`")
            case list() | dict():
                value_json = orjson.dumps(value, option=orjson.OPT_INDENT_2).decode()
                if len(value_json) > 100:
                    lines.append(f"**{key}**: `{value_json[:100]}...`")
                else:
                    lines.append(f"**{key}**: `{value_json}`")
            case _:
                lines.append(f"**{key}**: `{value}`")
    return "\n\n".join(lines)


def extract_text_from_message(msg: dict[str, Any]) -> str:
    message_data = msg.get("message", {})
    content_items = message_data.get("content", [])

    match content_items:
        case str():
            return content_items
        case list():
            text_items = [
                item.get("text", "") for item in content_items if isinstance(item, dict) and item.get("type") == "text"
            ]
            return " ".join(text_items)
        case _:
            return ""


def is_warmup_message(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if msg.get("type") == "user":
            text_content = extract_text_from_message(msg)
            return text_content.strip().lower() == "warmup"
    return False


def filter_warmup_pair(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_user_found = False
    first_assistant_found = False
    filtered_messages = []

    for msg in messages:
        msg_type = msg.get("type")

        if msg_type == "user" and not first_user_found:
            first_user_found = True
            continue

        if msg_type == "assistant" and first_user_found and not first_assistant_found:
            first_assistant_found = True
            continue

        if first_user_found:
            filtered_messages.append(msg)

    return filtered_messages


def find_last_share_command_index(messages: list[dict[str, Any]]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        if msg.get("type") == "compact_marker":
            continue

        if msg.get("type") != "user":
            continue

        message_data = msg.get("message", {})
        content = message_data.get("content", [])

        match content:
            case str():
                if (
                    "<command-name>/share</command-name>" in content
                    or "<command-name>/cc-plus:share</command-name>" in content
                ):
                    return i
            case list():
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if (
                            "<command-name>/share</command-name>" in text
                            or "<command-name>/cc-plus:share</command-name>" in text
                        ):
                            return i

    return None


def convert_transcript_to_markdown(
    transcript_path: Path, output_path: Path, session_id: str, usage_data: SessionUsageData | None
) -> None:
    if not transcript_path.exists():
        console.print(f"[red]Error: Transcript file not found: {transcript_path}[/red]")
        sys.exit(1)

    messages: list[dict[str, Any]] = []

    with transcript_path.open("rb") as f:
        for line in f:
            if line.strip():
                try:
                    data = orjson.loads(line)
                    if data.get("type") in ("user", "assistant", "compact_marker"):
                        messages.append(data)
                except orjson.JSONDecodeError:
                    continue

    if not messages:
        console.print("[yellow]No messages found in transcript[/yellow]")
        sys.exit(0)

    if is_warmup_message(messages):
        messages = filter_warmup_pair(messages)

    last_share_index = find_last_share_command_index(messages)
    if last_share_index is not None:
        messages = messages[:last_share_index]
        console.print(
            f"[yellow]📍 Truncating before /share command (excluded message #{last_share_index + 1})[/yellow]"
        )

    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    for msg in messages:
        msg_type = msg.get("type")
        if msg_type == "compact_marker":
            continue

        timestamp_str = msg.get("timestamp", "")
        timestamp_dt = parse_timestamp_to_datetime(timestamp_str)

        if timestamp_dt:
            if first_timestamp is None:
                first_timestamp = timestamp_dt
            last_timestamp = timestamp_dt

    total_session_time = 0.0
    if first_timestamp and last_timestamp:
        total_session_time = (last_timestamp - first_timestamp).total_seconds()

    class Turn(TypedDict):
        user_timestamp: datetime | None
        last_assistant_timestamp: datetime | None

    turns: list[Turn] = []
    current_turn: Turn | None = None

    for msg in messages:
        msg_type = msg.get("type")
        if msg_type == "compact_marker":
            continue

        timestamp_str = msg.get("timestamp", "")
        timestamp_dt = parse_timestamp_to_datetime(timestamp_str)

        match msg_type:
            case "user":
                if current_turn:
                    turns.append(current_turn)

                current_turn = Turn(user_timestamp=timestamp_dt, last_assistant_timestamp=None)

            case "assistant":
                if current_turn and timestamp_dt:
                    current_turn["last_assistant_timestamp"] = timestamp_dt

    if current_turn:
        turns.append(current_turn)

    llm_active_time_total = 0.0
    llm_idle_time_total = 0.0

    for i, turn in enumerate(turns):
        user_ts = turn["user_timestamp"]
        last_asst_ts = turn["last_assistant_timestamp"]

        if user_ts and last_asst_ts:
            active_duration = (last_asst_ts - user_ts).total_seconds()
            llm_active_time_total += active_duration

        if i + 1 < len(turns):
            next_turn = turns[i + 1]
            next_user_ts = next_turn["user_timestamp"]
            if last_asst_ts and next_user_ts:
                idle_duration = (next_user_ts - last_asst_ts).total_seconds()
                llm_idle_time_total += idle_duration

    md_lines = [
        "# 🤖 Claude Code Session Transcript",
        "",
        f"**Session ID**: `{session_id}`",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Messages**: {len(messages)}",
        "",
        "## 📊 Session Statistics",
        "",
    ]

    if usage_data:
        models_str = ", ".join(usage_data["modelsUsed"])
        md_lines.extend(
            [
                f"**Models Used**: {models_str}",
                "",
                "### Token Usage",
                "",
                f"- **Input Tokens**: {usage_data['inputTokens']:,}",
                f"- **Output Tokens**: {usage_data['outputTokens']:,}",
                f"- **Cache Creation**: {usage_data['cacheCreationTokens']:,}",
                f"- **Cache Read**: {usage_data['cacheReadTokens']:,}",
                f"- **Total Tokens**: {usage_data['totalTokens']:,}",
                "",
                "### 💰 Cost Estimate",
                "",
                f"- **Total Cost**: ${usage_data['totalCost']:.6f}",
            ]
        )

        if usage_data["totalTokens"] > 0 and usage_data["cacheReadTokens"] > 0:
            cache_hit_rate = (usage_data["cacheReadTokens"] / usage_data["totalTokens"]) * 100
            md_lines.append(f"- **Cache Hit Rate**: {cache_hit_rate:.2f}%")

        if usage_data["totalCost"] > 0:
            assistant_count = len([m for m in messages if m.get("type") == "assistant"])
            if assistant_count > 0:
                avg_cost_per_msg = usage_data["totalCost"] / assistant_count
                md_lines.append(f"- **Average Cost per Message**: ${avg_cost_per_msg:.6f}")
    else:
        md_lines.append("**Warning**: Session usage data not available from ccusage")

    if total_session_time > 0:
        md_lines.extend(["", "### ⏱️ Session Timeline", ""])
        md_lines.append(f"- **Total Session Time**: {format_duration(total_session_time)}")

        if llm_active_time_total > 0:
            md_lines.append(f"- **LLM Active Time**: {format_duration(llm_active_time_total)}")

        if llm_idle_time_total > 0:
            md_lines.append(f"- **LLM Idle Time**: {format_duration(llm_idle_time_total)}")

        if llm_active_time_total > 0 and total_session_time > 0:
            utilization = (llm_active_time_total / total_session_time) * 100
            md_lines.append(f"- **LLM Utilization**: {utilization:.1f}%")

    md_lines.extend(["", "---", ""])

    for i, msg in enumerate(messages, 1):
        msg_type = msg.get("type")

        if msg_type == "compact_marker":
            md_lines.extend(["---", "", "## 📦 [COMPACTED]", "", "---", ""])
            continue

        timestamp = parse_timestamp(msg.get("timestamp", ""))
        message_data = msg.get("message", {})
        role = message_data.get("role", msg_type)

        if role == "user":
            is_meta = msg.get("isMeta", False)
            content_items = message_data.get("content", [])

            if is_meta:
                continue
            else:
                match content_items:
                    case str():
                        escaped_content = escape_xml_tags(content_items)
                        quoted_lines = [f"> {line}" if line else ">" for line in escaped_content.split("\n")]
                        md_lines.extend(
                            [
                                f"## 💬 User #{i}",
                                f"**Time**: {timestamp}",
                                "",
                            ]
                        )
                        md_lines.extend(quoted_lines)
                        md_lines.append("")
                    case list():
                        text_items = [
                            item.get("text", "")
                            for item in content_items
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        if text_items:
                            md_lines.extend(
                                [
                                    f"## 💬 User #{i}",
                                    f"**Time**: {timestamp}",
                                    "",
                                ]
                            )
                            for text in text_items:
                                escaped_text = escape_xml_tags(text)
                                quoted_lines = [f"> {line}" if line else ">" for line in escaped_text.split("\n")]
                                md_lines.extend(quoted_lines)
                            md_lines.append("")

        else:
            md_lines.extend(
                [
                    f"## 🤖 Assistant #{i}",
                    f"**Time**: {timestamp}",
                    "",
                ]
            )

            content = message_data.get("content", [])
            match content:
                case str():
                    md_lines.extend([content, ""])
                case list():
                    for item in content:
                        if not isinstance(item, dict):
                            continue

                        item_type = item.get("type")

                        match item_type:
                            case "text":
                                text = item.get("text", "")
                                if text.strip():
                                    quoted_lines = [f"> {line}" if line else ">" for line in text.split("\n")]
                                    md_lines.extend(quoted_lines)
                                    md_lines.append("")

                            case "thinking":
                                thinking = item.get("thinking", "")
                                if thinking.strip():
                                    md_lines.append("> ")
                                    md_lines.append(">> 🧠 Thinking")
                                    thinking_lines = [f">> {line}" if line else ">>" for line in thinking.split("\n")]
                                    md_lines.extend(thinking_lines)
                                    md_lines.append(">")

                            case "tool_use":
                                tool_name = item.get("name", "unknown")
                                tool_input = item.get("input", {})

                                is_subagent = tool_name == "Task"
                                subagent_type = tool_input.get("subagent_type", "") if is_subagent else ""

                                if is_subagent:
                                    tool_display = f"🚀 Subagent: {subagent_type}"
                                else:
                                    tool_display = f"🔧 Tool: {tool_name}"

                                md_lines.extend(
                                    [
                                        "<details>",
                                        f"<summary>{tool_display}</summary>",
                                        "",
                                        format_tool_parameters(tool_input),
                                        "",
                                        "</details>",
                                        "",
                                    ]
                                )

        md_lines.extend(["---", ""])

    markdown_content = "\n".join(md_lines)

    while "\n---\n\n---\n" in markdown_content:
        markdown_content = markdown_content.replace("\n---\n\n---\n", "\n---\n")

    output_path.write_text(markdown_content, encoding="utf-8")
    console.print(f"[green]✅ Markdown saved to: {output_path}[/green]")

    if usage_data:
        console.print(f"\n[bold green]💰 Total Session Cost: ${usage_data['totalCost']:.6f}[/bold green]")


def convert_to_markdown(session_id: str) -> Path | None:
    transcript_path = find_transcript_path(session_id)
    if not transcript_path:
        console.print(f"[red]Error: Transcript not found for session {session_id}[/red]")
        return None

    merged_transcript_path = create_merged_transcript(session_id, transcript_path)
    if not merged_transcript_path:
        console.print("[red]Error: Failed to create merged transcript[/red]")
        return None

    backups = find_pre_compact_backups(session_id)
    if backups:
        console.print(f"[cyan]Found {len(backups)} pre-compact backup(s). Merging...[/cyan]")

    console.print("[cyan]Fetching session usage from ccusage...[/cyan]")
    usage_data = fetch_session_usage_from_ccusage(session_id)

    if usage_data:
        console.print("[green]✓ Session usage data loaded[/green]")
    else:
        console.print("[yellow]⚠ Could not fetch session usage data from ccusage[/yellow]")

    output_dir = Path("/tmp/claude-code-sessions")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"{session_id}-{date_str}.md"

    try:
        convert_transcript_to_markdown(merged_transcript_path, output_file, session_id, usage_data)
        return output_file
    except Exception as e:
        console.print(f"[red]Error during conversion: {e}[/red]")
        return None


def copy_to_clipboard(text: str) -> None:
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not copy to clipboard: {e}[/yellow]")


def main() -> None:
    if len(sys.argv) < 2:
        console.print("[red]Usage: share_session.py <query>[/red]")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    console.print(f"[cyan]Searching for session matching: {query}[/cyan]")

    todos_dir = Path.home() / ".claude" / "todos"
    session_id = search_todos(query, todos_dir)

    if not session_id:
        console.print(f"[red]No session found matching: {query}[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Found session: {session_id}[/green]")

    output_path = convert_to_markdown(session_id)

    if not output_path:
        sys.exit(1)

    copy_to_clipboard(str(output_path))

    console.print("\n[green]✅ Markdown saved to:[/green]")
    console.print(f"[bold]{output_path}[/bold]")
    console.print("\n[cyan]📋 The path has been copied to your clipboard.[/cyan]")

    print(output_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
