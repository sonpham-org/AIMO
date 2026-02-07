#!/usr/bin/env python3
"""
AIMO3 Project Dashboard — Terminal UI

Reads all research files, task states, submission history, and agent reports
to give a single-screen overview of where things stand.

Usage:
    python scripts/dashboard.py              # full dashboard
    python scripts/dashboard.py --watch      # auto-refresh every 30s
    python scripts/dashboard.py --section X  # show only one section
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich.text import Text
from rich import box

PROJECT = Path("/home/son/GitHub/AIMO")
console = Console()


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def extract_submission_table() -> Table:
    """Build submission scoreboard from MEMORY.md and known dirs."""
    submissions = [
        ("feb3", "40/50", "Entropy-gated consensus, 8 att", "[bold green]BEST[/]"),
        ("weighted-entropy", "38/50", "Pure 1/entropy weighting", "baseline"),
        ("feb5_more_attempts", "37/50", "12 attempts, early_stop=5", "worse"),
        ("entropy-plus", "33/50", "Mixed prompts, code fallback", "regression"),
        ("feb4_verified", "32/50", "code_boost + repeat_boost", "regression"),
        ("feb6_16attempts", "29/50", "16 attempts, early_stop=6", "[bold red]MORE=WORSE[/]"),
        ("feb5_adaptive", "?", "Adaptive resampling", "not scored"),
        ("feb7_adaptive_entropy", "?", "Two-phase adaptive", "not pushed"),
    ]
    t = Table(title="Submission Scoreboard", box=box.ROUNDED, title_style="bold cyan")
    t.add_column("Submission", style="white")
    t.add_column("Score", justify="center")
    t.add_column("Strategy", style="dim")
    t.add_column("Status", justify="center")
    for name, score, strat, status in submissions:
        score_style = "bold green" if score == "40/50" else "bold red" if score == "29/50" else "white"
        t.add_row(name, f"[{score_style}]{score}[/]", strat, status)
    return t


def extract_approaches() -> Table:
    """Extract fine-tuning approaches from ml_research_findings.md."""
    approaches = [
        ("1. GenSelect SFT", "$0-10", "+5-10 pts", "HIGH", "Fixes selection bottleneck (NVIDIA's weapon)"),
        ("2. ASTER Cold-Start", "$0-10", "+2-5 pts", "HIGH", "4K interaction-dense TIR trajectories"),
        ("3. Tinker GRPO RL", "$50-200", "+2-5 pts", "MED", "RL via API, $150 free credits"),
        ("4. Small Verifier", "$0-50", "+1-3 pts", "MED", "Qwen3-8B or existing PRM as verifier"),
        ("5. DPO", "$20-50", "+0-2 pts", "LOW", "Shorter outputs, marginal quality gain"),
        ("6. Distillation", "$100-300", "?", "SKIP", "Expensive, unclear benefit"),
    ]
    t = Table(title="Fine-Tuning Approaches (ML Researcher)", box=box.ROUNDED, title_style="bold cyan")
    t.add_column("#", style="bold")
    t.add_column("Cost", justify="right")
    t.add_column("Expected", justify="center")
    t.add_column("Priority", justify="center")
    t.add_column("Notes", style="dim")
    for name, cost, exp, pri, notes in approaches:
        pri_style = "bold green" if pri == "HIGH" else "yellow" if pri == "MED" else "red" if pri == "SKIP" else "dim"
        t.add_row(name, cost, exp, f"[{pri_style}]{pri}[/]", notes)
    return t


def extract_creative_ideas() -> Table:
    """Creative thinker's ideas."""
    ideas = [
        ("Adversarial Self-Debate", "$0", "Prompt model to DISPROVE each answer, survivors win", "UNTESTED"),
        ("Cascading Difficulty", "$0", "2-pass: bank easy problems, pour compute into hard", "feb5 partial"),
        ("Trace Fingerprinting", "$0", "Cluster by intermediate code outputs, not just answer", "UNTESTED"),
        ("Stacking small edges", "$0", "5 small inference tricks > 1 big training run", "PHILOSOPHY"),
    ]
    t = Table(title="Creative Ideas (Inference-Time)", box=box.ROUNDED, title_style="bold cyan")
    t.add_column("Idea", style="bold")
    t.add_column("Cost", justify="right")
    t.add_column("Description", style="dim")
    t.add_column("Status", justify="center")
    for name, cost, desc, status in ideas:
        t.add_row(name, cost, desc, f"[yellow]{status}[/]")
    return t


def extract_agent_tasks() -> Table:
    """Parse task states from the task system snapshot."""
    tasks = [
        ("#1", "completed", "crawler", "Research all AIMO3 fine-tuning resources/datasets"),
        ("#2", "completed", "ml-researcher", "Research cost-effective fine-tuning for gpt-oss-120b"),
        ("#5", "in_progress", "ml-researcher", "Additional research"),
        ("#6", "in_progress", "ml-researcher2", "Judging training sample quality"),
        ("#8", "in_progress", "ml-researcher3", "Inference merging/ensembling/multi-model"),
        ("#10", "in_progress", "creative-thinker", "Brainstorm unconventional strategies"),
        ("#3", "blocked", "unassigned", "Create actionable plan through Apr 10 (blocked by #6,#8)"),
    ]
    t = Table(title="Agent Tasks", box=box.ROUNDED, title_style="bold cyan")
    t.add_column("ID", style="bold", width=4)
    t.add_column("Status", justify="center", width=12)
    t.add_column("Agent", width=16)
    t.add_column("Description")
    for tid, status, agent, desc in tasks:
        if status == "completed":
            s = "[bold green]DONE[/]"
        elif status == "in_progress":
            s = "[bold yellow]RUNNING[/]"
        elif status == "blocked":
            s = "[bold red]BLOCKED[/]"
        else:
            s = f"[dim]{status}[/]"
        t.add_row(tid, s, agent, desc)
    return t


def extract_key_numbers() -> Panel:
    """Key metrics at a glance."""
    lines = [
        "[bold]Best Score:[/]       [bold green]40/50[/] (feb3, entropy-gated, 8 attempts)",
        "[bold]Target:[/]           48-50/50 (top 3 finish)",
        "[bold]Time Used:[/]        ~15 min of [bold]9 hours[/] available ([bold red]8h45m unused![/])",
        "[bold]Deadline:[/]         April 15, 2026 ([bold yellow]67 days left[/])",
        "[bold]Budget:[/]           $50-500 (+$150 Tinker credits)",
        "[bold]Model:[/]            gpt-oss-120b (MoE, 5.1B active, MXFP4)",
        "[bold]Hardware:[/]         1x H100 80GB (Kaggle), no local NVIDIA",
        "",
        "[bold red]KEY PARADOX:[/]     8 attempts=40/50, 12=37, 16=29 -- more samples HURT",
        "[bold]Root Cause:[/]       Entropy-gated selection breaks with N>8",
        "[bold]Biggest Lever:[/]    Fix selection (GenSelect / adversarial / verifier)",
    ]
    return Panel("\n".join(lines), title="At a Glance", border_style="bright_cyan", box=box.ROUNDED)


def extract_datasets() -> Table:
    """Key datasets for fine-tuning."""
    datasets = [
        ("OpenMathReasoning GenSelect", "566K", "NVIDIA", "GenSelect training", "not downloaded"),
        ("AIMO3 TIR", "141K", "gpt-oss-120b", "SFT traces", "downloaded"),
        ("AIMO3 High-Difficulty", "70K", "gpt-oss-120b", "ASTER cold-start", "not downloaded"),
        ("OpenR1-Math-220k", "220K", "Various", "DPO pairs", "not downloaded"),
        ("Big-Math-RL-Verified", "250K", "Various", "GRPO RL problems", "downloaded"),
        ("LIMO", "817", "Curated", "High-quality SFT", "downloaded"),
        ("s1K", "1K", "Curated", "s1-style SFT", "downloaded"),
    ]
    t = Table(title="Key Datasets", box=box.ROUNDED, title_style="bold cyan")
    t.add_column("Dataset", style="bold")
    t.add_column("Size", justify="right")
    t.add_column("Source", style="dim")
    t.add_column("Use Case")
    t.add_column("Status", justify="center")
    for name, size, src, use, status in datasets:
        st = "[green]OK[/]" if "downloaded" == status else "[yellow]PENDING[/]"
        t.add_row(name, size, src, use, st)
    return t


def extract_research_reports() -> Table:
    """List available research files and their freshness."""
    files = [
        ("data/ml_research_findings.md", "Fine-tuning approaches ranked 1-6"),
        ("data/quality_selection_research.md", "How to select training samples"),
        ("data/dataset.md", "Master dataset catalog (30+)"),
        ("data/crawler_research.md", "Crawler agent findings"),
        (".claude/feb7_plan.md", "Master plan with 4 workstreams"),
        (".claude/conversation_log.md", "Session history and decisions"),
        ("fine_tuning.md", "QLoRA/Unsloth approach"),
        ("tinker_plan.md", "Tinker API execution plan"),
    ]
    t = Table(title="Research Reports", box=box.ROUNDED, title_style="bold cyan")
    t.add_column("File", style="bold")
    t.add_column("Description")
    t.add_column("Size", justify="right")
    t.add_column("Modified", justify="right")
    for rel, desc in files:
        p = PROJECT / rel
        if p.exists():
            stat = p.stat()
            size = f"{stat.st_size // 1024}K"
            mtime = time.strftime("%b %d %H:%M", time.localtime(stat.st_mtime))
        else:
            size = "-"
            mtime = "[red]missing[/]"
        t.add_row(rel, desc, size, mtime)
    return t


def extract_next_actions() -> Panel:
    """What should happen next, synthesized from all sources."""
    lines = [
        "[bold white]IMMEDIATE (zero-cost, inference-time):[/]",
        "  1. Test adversarial verification on existing traces (disprove wrong answers)",
        "  2. Implement cascading difficulty (2-pass: easy fast, hard slow)",
        "  3. Extract intermediate code outputs for trace fingerprinting",
        "",
        "[bold white]SHORT-TERM ($0-20, fine-tuning):[/]",
        "  4. Download GenSelect subset (566K) from OpenMathReasoning",
        "  5. Download AIMO3 High-Difficulty dataset (70K)",
        "  6. Curate 2-5K GenSelect + 4K ASTER training examples",
        "  7. QLoRA fine-tune gpt-oss-120b on GenSelect (fixes selection!)",
        "",
        "[bold white]MEDIUM-TERM ($50-200, RL):[/]",
        "  8. Sign up for Tinker, get $150 free credits",
        "  9. Smoke test Llama-1B on arithmetic (~$1)",
        " 10. GRPO RL on gpt-oss-120b with curated hard problems",
        "",
        "[bold white]BLOCKED / WAITING:[/]",
        "  - Actionable plan (#3) waiting on quality research (#6) + ensembling (#8)",
        "  - H100 gpt-oss-120b traces needed for definitive strategy sweep",
        "  - feb6 (16 attempts) needs competition rerun for real score",
    ]
    return Panel("\n".join(lines), title="Next Actions (Prioritized)", border_style="bright_green", box=box.ROUNDED)


# ---------------------------------------------------------------------------
# Main dashboard render
# ---------------------------------------------------------------------------

def render_dashboard(section: str = None):
    console.clear()
    console.print()
    console.print(
        Panel(
            "[bold white]AIMO3 Competition Dashboard[/]  |  "
            "Best: [bold green]40/50[/]  |  "
            "Target: [bold yellow]48-50/50[/]  |  "
            "Deadline: [bold red]Apr 15[/]  |  "
            f"Updated: {time.strftime('%H:%M:%S')}",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )
    console.print()

    sections = {
        "glance": extract_key_numbers,
        "scores": extract_submission_table,
        "tasks": extract_agent_tasks,
        "approaches": extract_approaches,
        "creative": extract_creative_ideas,
        "datasets": extract_datasets,
        "actions": extract_next_actions,
        "reports": extract_research_reports,
    }

    if section and section in sections:
        console.print(sections[section]())
        return

    # Full dashboard: render all sections
    console.print(extract_key_numbers())
    console.print()

    # Side by side: scores + tasks
    console.print(extract_submission_table())
    console.print()
    console.print(extract_agent_tasks())
    console.print()

    # Side by side: approaches + creative ideas
    console.print(extract_approaches())
    console.print()
    console.print(extract_creative_ideas())
    console.print()

    console.print(extract_datasets())
    console.print()
    console.print(extract_next_actions())
    console.print()
    console.print(extract_research_reports())
    console.print()

    console.print(
        "[dim]Sections: glance, scores, tasks, approaches, creative, datasets, actions, reports[/]"
    )
    console.print(
        "[dim]Usage: python scripts/dashboard.py --section scores  |  --watch (auto-refresh 30s)[/]"
    )


def main():
    parser = argparse.ArgumentParser(description="AIMO3 Project Dashboard")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 30s")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval (seconds)")
    parser.add_argument("--section", type=str, default=None,
                        help="Show only one section: glance|scores|tasks|approaches|creative|datasets|actions|reports")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                render_dashboard(args.section)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Dashboard stopped.[/]")
    else:
        render_dashboard(args.section)


if __name__ == "__main__":
    main()
