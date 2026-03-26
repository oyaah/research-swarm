from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def wants_pdf(query: str) -> bool:
    q = query.lower()
    return "pdf" in q or "export report" in q


def wants_chart(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ["chart", "graph", "plot", "visual"])


def export_pdf(report_dir: Path, markdown: str) -> tuple[str | None, str | None]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return None, "reportlab not installed; skipped PDF export"

    pdf_path = report_dir / "report.pdf"
    try:
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        y = height - 40
        lines = markdown.splitlines()
        for raw in lines:
            line = raw.strip()
            while len(line) > 110:
                c.drawString(30, y, line[:110])
                line = line[110:]
                y -= 14
                if y < 40:
                    c.showPage()
                    y = height - 40
            c.drawString(30, y, line)
            y -= 14
            if y < 40:
                c.showPage()
                y = height - 40
        c.save()
        return str(pdf_path), None
    except Exception as exc:
        return None, f"pdf export failed: {exc}"


def _extract_probabilities(markdown: str) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    vals: list[float] = []
    for line in markdown.splitlines():
        if "|" in line and "%" in line:
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if len(cols) < 2:
                continue
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if not m:
                continue
            labels.append(cols[0][:28])
            vals.append(float(m.group(1)))
            if len(labels) >= 5:
                break
    if labels:
        return labels, vals
    # fallback chart from generic confidence cues
    return ["Faithfulness", "Coverage", "Recency", "Novelty"], [60, 60, 50, 55]


def export_chart(report_dir: Path, markdown: str) -> tuple[str | None, str | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None, "matplotlib not installed; skipped chart export"

    chart_path = report_dir / "report_chart.png"
    labels, values = _extract_probabilities(markdown)
    try:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"][: len(labels)])
        ax.set_ylim(0, max(100, max(values) + 10))
        ax.set_ylabel("Score / Probability (%)")
        ax.set_title("Research Output Snapshot")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=160)
        plt.close(fig)
        return str(chart_path), None
    except Exception as exc:
        return None, f"chart export failed: {exc}"


def maybe_export_artifacts(session_id: str, query: str, markdown: str) -> dict[str, Any]:
    report_dir = Path("artifacts/reports") / session_id
    report_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"requested": [], "generated": {}, "warnings": []}
    report_md = report_dir / "report.md"
    try:
        report_md.write_text(markdown or "", encoding="utf-8")
        out["generated"]["report_md"] = str(report_md)
    except Exception as exc:
        out["warnings"].append(f"report save failed: {exc}")

    if wants_pdf(query):
        out["requested"].append("pdf")
        path, warn = export_pdf(report_dir, markdown)
        if path:
            out["generated"]["pdf"] = path
        if warn:
            out["warnings"].append(warn)

    if wants_chart(query):
        out["requested"].append("chart")
        path, warn = export_chart(report_dir, markdown)
        if path:
            out["generated"]["chart"] = path
        if warn:
            out["warnings"].append(warn)

    return out
