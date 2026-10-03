"""Compact experiment history shown to the agent each experiment (never code)."""

from __future__ import annotations


def _clip(text: str | None, n: int) -> str:
    text = (text or "").replace("\n", " ").replace("|", "/").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def render_history(records: list[dict], budget: int, used: int) -> str:
    ok = [r for r in records if r["status"] == "ok" and r["mae"] is not None]
    best = min(ok, key=lambda r: r["mae"]) if ok else None
    lines = [
        "| id | parent | hypothesis | mutation | val MAE | n_feat | status |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for r in records:
        mae = f"{r['mae']:.3f}" if r["mae"] is not None else "—"
        status = r["status"] if r["status"] == "ok" else f"{r['status']}: {_clip(r.get('error'), 120)}"
        lines.append(
            f"| {r['experiment_id']} | {r['parent_id'] or '—'} | {_clip(r['hypothesis'], 90)} | "
            f"{_clip(r['transformation_description'], 110)} | {mae} | {r['n_features'] or '—'} | {status} |"
        )
    head = [f"Experiments used: {used} of {budget}. Remaining: {budget - used}."]
    if best:
        head.append(f"Best so far: {best['experiment_id']} with validation MAE {best['mae']:.3f}.")
    return "\n".join(head) + "\n\n" + "\n".join(lines)
