from __future__ import annotations

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "runs"
    rows = []
    for name in (
        "ablation-plain",
        "ablation-no-graph",
        "ablation-no-fields",
        "ablation-full",
        "ablation-hybrid",
        "ablation-hybrid-sharp",
        "ablation-hybrid-anchor",
        "ablation-hybrid-fast",
        "baseline-simple-r18",
    ):
        history_path = root / name / "history.json"
        if not history_path.exists():
            rows.append({"run": name, "status": "pending"})
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best = max(history, key=lambda item: item["validation"]["pck@0.1"])
        rows.append({
            "run": name,
            "status": "complete" if len(history) == 8 else f"{len(history)}/8 epochs",
            "best_epoch": best["epoch"],
            "pck@0.05": best["validation"]["pck@0.05"],
            "pck@0.1": best["validation"]["pck@0.1"],
            "pck@0.15": best["validation"]["pck@0.15"],
            "normalized_mean_error": best["validation"]["normalized_mean_error"],
            "seconds_per_epoch": best["seconds"],
        })
    output = root / "ablation-summary.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
