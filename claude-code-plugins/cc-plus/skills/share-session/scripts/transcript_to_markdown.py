#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "~=3.12"
# dependencies = [
#     "orjson",
#     "rich",
#     "typer",
#     "httpx",
# ]
# ///

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import httpx  # pyright: ignore[reportMissingImports]
import orjson  # pyright: ignore[reportMissingImports]
import typer  # pyright: ignore[reportMissingImports]
from rich.console import Console  # pyright: ignore[reportMissingImports]
from rich.markdown import Markdown  # pyright: ignore[reportMissingImports]

app = typer.Typer()
console = Console()

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"
)


class ModelPricing(TypedDict):
    input_cost_per_token: NotRequired[float]
    output_cost_per_token: NotRequired[float]
    cache_creation_input_token_cost: NotRequired[float]
    cache_read_input_token_cost: NotRequired[float]
    litellm_provider: str
    mode: str


def escape_xml_tags(text: str) -> str:
    """Escape XML/HTML tags so they display in markdown."""
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


def format_tool_result(result: dict[str, Any]) -> str:
    content = result.get("content", "")
    is_error = result.get("is_error", False)

    if is_error:
        return f"❌ **Error**:\n```\n{content}\n```"

    match content:
        case str():
            return f"```\n{content}\n```"
        case _:
            return f"```json\n{orjson.dumps(content, option=orjson.OPT_INDENT_2).decode()}\n```"


def build_tool_map(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tool_map: dict[str, dict[str, Any]] = {}

    for msg in messages:
        if msg.get("type") == "user":
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tool_use_id = item.get("tool_use_id")
                        if tool_use_id:
                            tool_map[tool_use_id] = item

    return tool_map


def fetch_pricing_data() -> dict[str, ModelPricing]:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(LITELLM_PRICING_URL)
        response.raise_for_status()
        data = orjson.loads(response.content)
        if "sample_spec" in data:
            del data["sample_spec"]
        return data


def calculate_message_cost(
    usage: dict[str, Any], model: str, pricing_data: dict[str, ModelPricing]
) -> tuple[float, dict[str, int]]:
    pricing = pricing_data.get(model)
    if not pricing:
        return 0.0, {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
    cache_read_tokens = usage.get("cache_read_input_tokens", 0)

    input_rate = pricing.get("input_cost_per_token", 0.0)
    output_rate = pricing.get("output_cost_per_token", 0.0)
    cache_creation_rate = pricing.get("cache_creation_input_token_cost", 0.0)
    cache_read_rate = pricing.get("cache_read_input_token_cost", 0.0)

    total_cost = (
        input_tokens * input_rate
        + output_tokens * output_rate
        + cache_creation_tokens * cache_creation_rate
        + cache_read_tokens * cache_read_rate
    )

    token_breakdown = {
        "input": input_tokens,
        "output": output_tokens,
        "cache_creation": cache_creation_tokens,
        "cache_read": cache_read_tokens,
    }

    return total_cost, token_breakdown


def convert_transcript_to_markdown(transcript_path: Path, output_path: Path | None = None) -> None:
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

    build_tool_map(messages)

    console.print("[cyan]Fetching pricing data...[/cyan]")
    try:
        pricing_data = fetch_pricing_data()
        console.print("[green]✓ Pricing data loaded[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not fetch pricing data: {e}[/yellow]")
        pricing_data = {}

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    models_used: dict[str, int] = {}

    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    last_user_timestamp: datetime | None = None
    llm_time_seconds = 0.0
    llm_started = False

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

        if msg_type == "user":
            last_user_timestamp = timestamp_dt
            llm_started = False
        elif msg_type == "assistant":
            if last_user_timestamp and timestamp_dt and not llm_started:
                llm_duration = (timestamp_dt - last_user_timestamp).total_seconds()
                llm_time_seconds += llm_duration
                llm_started = True

            message_data = msg.get("message", {})
            usage = message_data.get("usage")
            if usage:
                model = message_data.get("model", "unknown")
                models_used[model] = models_used.get(model, 0) + 1

                cost, breakdown = calculate_message_cost(usage, model, pricing_data)
                total_cost += cost
                total_input_tokens += breakdown.get("input", 0)
                total_output_tokens += breakdown.get("output", 0)
                total_cache_creation_tokens += breakdown.get("cache_creation", 0)
                total_cache_read_tokens += breakdown.get("cache_read", 0)

    total_tokens = total_input_tokens + total_output_tokens + total_cache_creation_tokens + total_cache_read_tokens

    total_session_time = 0.0
    if first_timestamp and last_timestamp:
        total_session_time = (last_timestamp - first_timestamp).total_seconds()

    md_lines = [
        "# 🤖 Claude Code Session Transcript",
        "",
        f"**Session ID**: `{messages[0].get('sessionId', 'unknown')}`",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Messages**: {len(messages)}",
        "",
        "## 📊 Session Statistics",
        "",
        f"**Models Used**: {', '.join(f'{model} ({count})' for model, count in models_used.items())}",
        "",
        "### Token Usage",
        "",
        f"- **Input Tokens**: {total_input_tokens:,}",
        f"- **Output Tokens**: {total_output_tokens:,}",
        f"- **Cache Creation**: {total_cache_creation_tokens:,}",
        f"- **Cache Read**: {total_cache_read_tokens:,}",
        f"- **Total Tokens**: {total_tokens:,}",
        "",
        "### 💰 Cost Estimate",
        "",
        f"- **Total Cost**: ${total_cost:.6f}",
    ]

    if total_tokens > 0 and total_cache_read_tokens > 0:
        cache_hit_rate = (total_cache_read_tokens / total_tokens) * 100
        md_lines.append(f"- **Cache Hit Rate**: {cache_hit_rate:.2f}%")

    if total_cost > 0:
        assistant_count = len([m for m in messages if m.get("type") == "assistant"])
        if assistant_count > 0:
            avg_cost_per_msg = total_cost / assistant_count
            md_lines.append(f"- **Average Cost per Message**: ${avg_cost_per_msg:.6f}")

    if total_session_time > 0:
        md_lines.extend(["", "### ⏱️ Session Timeline", ""])
        md_lines.append(f"- **Total Session Time**: {format_duration(total_session_time)}")
        md_lines.append(f"- **LLM Active Time**: {format_duration(llm_time_seconds)}")

        wait_time = total_session_time - llm_time_seconds
        if wait_time > 0:
            md_lines.append(f"- **Wait Time**: {format_duration(wait_time)}")

        if total_session_time > 0:
            utilization = (llm_time_seconds / total_session_time) * 100
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

    if output_path:
        output_path.write_text(markdown_content, encoding="utf-8")
        console.print(f"[green]✅ Markdown saved to: {output_path}[/green]")
    else:
        default_output = transcript_path.with_suffix(".md")
        default_output.write_text(markdown_content, encoding="utf-8")
        console.print(f"[green]✅ Markdown saved to: {default_output}[/green]")

    console.print(f"\n[bold green]💰 Total Session Cost: ${total_cost:.6f}[/bold green]")
    console.print("\n[cyan]Preview:[/cyan]")
    console.print(Markdown(markdown_content[:1000] + "\n\n... (truncated)"))


@app.command()
def main(
    transcript: Path = typer.Argument(..., help="Path to transcript JSONL file"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output markdown file path"),
    preview: bool = typer.Option(False, "--preview", "-p", help="Preview in terminal instead of saving"),
) -> None:
    if preview:
        console.print("[yellow]Preview mode - content will not be saved[/yellow]\n")

    convert_transcript_to_markdown(transcript, output)


if __name__ == "__main__":
    app()
