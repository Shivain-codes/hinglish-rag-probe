"""Publication-friendly figures and a self-contained escaped HTML report."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any


def render_report(payload: dict[str, Any], output: str | Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install hinglish-rag-probe[reports] for report generation") from exc
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = payload.get("reports") or {payload["metadata"]["retriever"]["name"]: payload}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    images = []
    for name, metric in (
        ("quality_by_condition", "ndcg"),
        ("recall", "recall"),
        ("mrr", "mrr"),
        ("degradation", "degradation"),
        ("accuracy_latency", "latency"),
        ("bootstrap_intervals", "ci"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        if metric in ("ndcg", "recall", "mrr"):
            conditions = sorted({c for r in reports.values() for c in r["summary"]})
            for i, (method, report) in enumerate(reports.items()):
                points = [
                    (j, report["summary"][c][metric])
                    for j, c in enumerate(conditions)
                    if c in report["summary"] and report["summary"][c][metric] is not None
                ]
                width = 0.8 / len(reports)
                ax.bar(
                    [x + (i - (len(reports) - 1) / 2) * width for x, _ in points],
                    [y for _, y in points],
                    width,
                    label=method,
                )
            ax.set_xticks(range(len(conditions)), conditions, rotation=20, ha="right")
            ax.set_ylabel(f"{metric.upper()}@K")
            ax.set_ylim(0, 1.05)
            ax.legend()
        elif metric == "latency":
            from matplotlib.lines import Line2D

            conditions = sorted({c for r in reports.values() for c in r["summary"]})
            markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
            colors = plt.get_cmap("tab10")
            for method_index, (_method, report) in enumerate(reports.items()):
                for condition, row in report["summary"].items():
                    if row["ndcg"] is not None:
                        marker = markers[conditions.index(condition) % len(markers)]
                        ax.scatter(
                            row["mean_latency_seconds"] * 1000,
                            row["ndcg"],
                            marker=marker,
                            color=colors(method_index % 10),
                            s=65,
                            edgecolor="white",
                            linewidth=0.5,
                        )
            handles = [
                Line2D([], [], color=colors(i % 10), marker="o", linestyle="", label=method)
                for i, method in enumerate(reports)
            ]
            handles += [
                Line2D(
                    [],
                    [],
                    color="grey",
                    marker=markers[i % len(markers)],
                    linestyle="",
                    label=c.replace("_", " "),
                )
                for i, c in enumerate(conditions)
            ]
            ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            ax.set_xscale("log")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Mean query latency (ms, log scale; instrumented)")
            ax.set_ylabel("NDCG@K")
        else:
            comparisons = payload.get("comparisons", []) if metric == "ci" else []
            rows = [
                (f"{c['candidate']} − {c['baseline']} / {c.get('condition', '')}", c)
                for c in comparisons
                if c["metric"] == "ndcg"
            ]
            if not rows:
                rows = [
                    (f"{method}: {c['candidate']} vs English", c)
                    for method, report in reports.items()
                    for c in report["comparisons"]
                    if c["metric"] == "ndcg"
                ]
            if metric == "degradation":
                rows = [
                    (label, c) for label, c in rows if c.get("relative_degradation") is not None
                ]
            if rows:
                fig.set_size_inches(10, max(4.8, len(rows) * 0.34 + 1.4))
                for i, (_, c) in enumerate(rows):
                    value = (
                        c["relative_degradation"] if metric == "degradation" else c["difference"]
                    )
                    ax.scatter(value, i)
                    if metric == "ci":
                        ax.hlines(i, c["ci_low"], c["ci_high"], linewidth=2)
                ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=8)
                ax.axvline(0, color="grey", linewidth=0.7)
            else:
                ax.text(
                    0.5, 0.5, "No eligible paired comparisons", ha="center", transform=ax.transAxes
                )
            ax.set_xlabel(
                "Relative NDCG loss from English"
                if metric == "degradation"
                else "Paired NDCG difference with percentile bootstrap CI"
            )
        k_values = {r["metadata"]["k"] for r in reports.values()}
        k_label = ", ".join(str(k) for k in sorted(k_values))
        ax.set_title(name.replace("_", " ").title() + f" (K={k_label})")
        if any(r["metadata"].get("dataset", {}).get("synthetic") for r in reports.values()):
            fig.suptitle(
                "SYNTHETIC DEMONSTRATION · not scientific evidence", fontsize=9, color="#805000"
            )
        for ext in ("svg", "pdf", "png"):
            fig.savefig(output / f"{name}.{ext}", dpi=180)
        plt.close(fig)
        image = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        images.append(f'<figure><img alt="{name}" src="data:image/png;base64,{image}"></figure>')
    rows_html = []
    for method, report in reports.items():
        for condition, row in report["summary"].items():
            cells = [method, condition, str(row["n_judged"])] + [
                "N/A" if row[m] is None else f"{row[m]:.4f}"
                for m in ("precision", "recall", "mrr", "map", "ndcg")
            ]
            rows_html.append(
                "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>"
            )
    metadata = {name: report["metadata"] for name, report in reports.items()}
    page = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>HinglishRAGProbe report</title>
<style>body{font:16px system-ui;margin:40px auto;max-width:1100px;padding:0 24px;color:#162a3b}
img{width:100%}
table{border-collapse:collapse;width:100%}
td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:20px}
h1{color:#174c62}</style>
<h1>HinglishRAGProbe</h1>
<p>Retrieval robustness experiment report</p>
<p>Dataset provenance below governs interpretation. 
Synthetic demonstrations are not scientific evidence.
Missing judgments are excluded; judged-empty queries score zero. 
Differences use paired original queries.
Intervals and p-values are exploratory and unadjusted for multiple comparisons. 
Zero-baseline relative
loss is undefined. Latency includes instrumentation and the first query; 
Python allocation peaks exclude
most native and GPU memory. Automatic variants are not assumed semantically equivalent.</p>
<table>
<thead>
<tr>
<th>Method</th>
<th>Condition</th>
<th>Judged</th>
<th>P@K</th>
<th>R@K</th>
<th>MRR@K</th>
<th>MAP@K</th>
<th>NDCG@K</th>
</tr>
</thead>
<tbody>"""
    page += "".join(rows_html) + "</tbody></table>" + "".join(images)
    page += (
        "<h2>Metadata</h2><pre>"
        + html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))
        + "</pre>"
    )
    page += (
        "<h2>Paired statistics</h2><pre>"
        + html.escape(
            json.dumps(
                payload.get("comparisons", [])
                + [c for r in reports.values() for c in r["comparisons"]],
                indent=2,
            )
        )
        + "</pre></html>"
    )
    target = output / "index.html"
    target.write_text(page, encoding="utf-8")
    return target
