"""Plot benchmark + training results and write a Markdown results table.

    uv run python scripts/visualize_results.py
    uv run python scripts/visualize_results.py --benchmark outputs/benchmark --out assets/results

Reads only files produced by evaluate.py / train_bc.py; it never invents numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Validated categorical order (blue, orange, aqua, yellow, magenta, green) - colour follows the config.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {
    "A_oracle_scripted": "A  Oracle + scripted",
    "B_vision_scripted": "B  Vision + scripted",
    "C_vision_bc": "C  Vision + BC policy",
    "C2_vision_bc_norecovery": "C2 BC, recovery OFF",
    "D_vision_scripted_faults": "D  Faults, recovery ON",
    "E_vision_scripted_faults_norecovery": "E  Faults, recovery OFF",
}


def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)


def grouped_bars(
    ax, groups: list[str], series: dict[str, list[float]], colors: dict[str, str], pct=True
):
    n = len(series)
    width = 0.8 / max(1, n)
    for i, (name, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(
            xs, vals, width * 0.92, color=colors[name], label=LABELS.get(name, name), zorder=2
        )
        for b, v in zip(bars, vals, strict=True):
            text = f"{v * 100:.0f}%" if pct else f"{v:.2f}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.015,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=INK)
    if pct:
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark",
        nargs="+",
        default=["outputs/benchmark"],
        help="one or more evaluate.py output directories (results are merged)",
    )
    p.add_argument("--training-log", default="outputs/checkpoints/training_log.json")
    p.add_argument("--out", default="assets/results")
    args = p.parse_args()

    bdirs = [ROOT / b for b in args.benchmark]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    metas: list[dict] = []
    for bdir in bdirs:
        summary_path = bdir / "summary.json"
        if not summary_path.exists():
            print(f"{summary_path} not found - run scripts/evaluate.py first")
            return 2
        data = json.loads(summary_path.read_text())
        results.update(data["results"])
        metas.append(data["meta"])
    meta = metas[0]
    if any(m["seeds"] != meta["seeds"] for m in metas):
        print("warning: merged benchmark directories used different seed sets")
    results = dict(sorted(results.items()))
    names = list(results)
    colors = {n: SERIES[i % len(SERIES)] for i, n in enumerate(LABELS) if n in names}
    for i, n in enumerate(names):
        colors.setdefault(n, SERIES[i % len(SERIES)])
    written = []

    # 1. Main comparison A/B/C
    main_cfgs = [n for n in names if n[0] in "ABC"]
    if main_cfgs:
        fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
        metrics = [
            ("task_success_rate", "Task success"),
            ("placement_success_rate", "Placement success"),
            ("pick_success_rate", "Pick success / attempt"),
        ]
        grouped_bars(
            ax,
            [m[1] for m in metrics],
            {n: [results[n][m[0]] for m in metrics] for n in main_cfgs},
            colors,
        )
        _style(
            ax,
            f"Policy comparison ({results[main_cfgs[0]]['episodes']} episodes each, fixed seeds)",
        )
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(main_cfgs),
            labelcolor=INK,
        )
        fig.tight_layout()
        fig.savefig(out / "benchmark_comparison.png", dpi=150)
        written.append("benchmark_comparison.png")
        plt.close(fig)

        # 2. Per-object placement success
        objs = list(results[main_cfgs[0]]["per_object"])
        fig, ax = plt.subplots(figsize=(9, 4.0), facecolor=SURFACE)
        grouped_bars(
            ax,
            objs,
            {
                n: [results[n]["per_object"][o]["placement_success_rate"] or 0.0 for o in objs]
                for n in main_cfgs
            },
            colors,
        )
        _style(ax, "Per-object placement success")
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(main_cfgs),
            labelcolor=INK,
        )
        fig.tight_layout()
        fig.savefig(out / "per_object_success.png", dpi=150)
        written.append("per_object_success.png")
        plt.close(fig)

    # 3. Recovery ablation D vs E
    abl = [n for n in names if n[0] in "DE"]
    if abl:
        fig, ax = plt.subplots(figsize=(7, 4.0), facecolor=SURFACE)
        metrics = [
            ("task_success_rate", "Task success"),
            ("placement_success_rate", "Placement success"),
        ]
        abl_colors = {n: SERIES[i] for i, n in enumerate(abl)}
        grouped_bars(
            ax,
            [m[1] for m in metrics],
            {n: [results[n][m[0]] for m in metrics] for n in abl},
            abl_colors,
        )
        rec = ", ".join(
            f"{LABELS.get(n, n).split()[0]}: {results[n]['avg_recovery_attempts']:.2f}" for n in abl
        )
        _style(ax, f"Failure recovery under injected faults (avg recoveries/episode - {rec})")
        ax.title.set_fontsize(10.5)
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(abl),
            labelcolor=INK,
        )
        fig.tight_layout()
        fig.savefig(out / "recovery_ablation.png", dpi=150)
        written.append("recovery_ablation.png")
        plt.close(fig)

    # 4. Episode duration distribution
    rows = []
    for bdir in bdirs:
        if (bdir / "episodes.csv").exists():
            rows += list(csv.DictReader((bdir / "episodes.csv").open()))
    if rows:
        fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
        data_by = [[float(r["sim_time_s"]) for r in rows if r["config"] == n] for n in names]
        bp = ax.boxplot(
            data_by, patch_artist=True, widths=0.5, medianprops={"color": INK, "linewidth": 1.5}
        )
        for patch, n in zip(bp["boxes"], names, strict=True):
            patch.set_facecolor(colors[n])
            patch.set_edgecolor(SURFACE)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([LABELS.get(n, n).split("  ")[0] for n in names], color=INK)
        _style(ax, "Episode duration (simulated seconds, 20 Hz control)", "seconds")
        fig.tight_layout()
        fig.savefig(out / "episode_duration.png", dpi=150)
        written.append("episode_duration.png")
        plt.close(fig)

    # 5. BC training curves (two single-axis panels)
    tlog = ROOT / args.training_log
    if tlog.exists():
        h = json.loads(tlog.read_text())["history"]
        ep = [r["epoch"] for r in h]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
        a1.plot(ep, [r["train_loss"] for r in h], color=SERIES[0], lw=2, label="train")
        a1.plot(
            ep,
            [r["val_loss"] for r in h],
            color=SERIES[1],
            lw=2,
            label="validation (held-out episodes)",
        )
        _style(a1, "BC loss (MSE translation + BCE gripper)")
        a1.set_xlabel("epoch", color=INK2)
        a1.legend(frameon=False, labelcolor=INK)
        a2.plot(ep, [r["val_grip_acc"] for r in h], color=SERIES[0], lw=2)
        _style(a2, "Validation gripper accuracy")
        a2.set_xlabel("epoch", color=INK2)
        fig.tight_layout()
        fig.savefig(out / "bc_training.png", dpi=150)
        written.append("bc_training.png")
        plt.close(fig)

    # Markdown table
    cols = [
        ("task_success_rate", "Task success", "pct"),
        ("placement_success_rate", "Placement success", "pct"),
        ("pick_success_rate", "Pick success / attempt", "pct"),
        ("avg_recovery_attempts", "Avg recoveries", "f2"),
        ("collision_rate", "Collision rate", "pct"),
        ("avg_manipulation_actions", "Avg pick-place actions", "f2"),
        ("avg_episode_sim_time_s", "Avg duration (sim s)", "f1"),
        ("avg_episode_wall_time_s", "Avg wall time (s)", "f1"),
        ("belief_ground_truth_agreement", "Self-check agreement", "pct"),
    ]
    fmt = {
        "pct": lambda v: f"{v * 100:.1f}%",
        "f2": lambda v: f"{v:.2f}",
        "f1": lambda v: f"{v:.1f}",
    }
    lines = [
        "| Config | Episodes | " + " | ".join(c[1] for c in cols) + " |",
        "|---" * (len(cols) + 2) + "|",
    ]
    for n in names:
        s = results[n]
        lines.append(
            f"| {LABELS.get(n, n).strip()} | {s['episodes']} | "
            + " | ".join(fmt[c[2]](s[c[0]]) for c in cols)
            + " |"
        )
    lines.append("")
    lines.append("Per-object placement success:")
    lines.append("")
    objs = list(results[names[0]]["per_object"])
    lines.append("| Config | " + " | ".join(objs) + " |")
    lines.append("|---" * (len(objs) + 1) + "|")
    for n in names:
        po = results[n]["per_object"]
        lines.append(
            f"| {LABELS.get(n, n).strip()} | "
            + " | ".join(f"{(po[o]['placement_success_rate'] or 0) * 100:.0f}%" for o in objs)
            + " |"
        )
    lines.append("")
    commits = ", ".join(sorted({str(m.get("git_commit")) for m in metas}))
    v = meta["versions"]
    lines.append(
        f"Seeds {meta['seeds'][0]}..{meta['seeds'][-1]} · commit(s) {commits} · "
        f"robosuite {v['robosuite']}, MuJoCo {v['mujoco']}, torch {v['torch']}, Python {v['python']}"
    )
    (out / "results_table.md").write_text("\n".join(lines) + "\n")
    written.append("results_table.md")
    print("\n".join(lines))
    print(f"\nwrote {', '.join(written)} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
