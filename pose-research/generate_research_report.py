from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def best_history(name: str):
    path = RUNS / name / "history.json"
    if not path.exists():
        return None
    history = load(path)
    return max(history, key=lambda row: row["validation"]["pck@0.1"])


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    development_runs = (
        ("plain", "ablation-plain"),
        ("no graph", "ablation-no-graph"),
        ("no fields", "ablation-no-fields"),
        ("original full", "ablation-full"),
        ("hybrid", "ablation-hybrid"),
        ("hybrid sharp", "ablation-hybrid-sharp"),
        ("hybrid anchor", "ablation-hybrid-anchor"),
        ("hybrid fast", "ablation-hybrid-fast"),
        ("SimpleBaseline R18", "baseline-simple-r18"),
        ("KineRes plain", "kineres-plain"),
        ("KineRes graph (3 layers)", "kineres-graph"),
        ("KineRes graph (1 layer)", "kineres-graph1"),
    )
    lines = [
        "# Adult pose architecture research report",
        "",
        "Generated from local checkpoints and JSON benchmark artifacts. All development models were trained from random initialization on the same fixed 24,000/1,800 COCO person-crop protocol unless stated otherwise.",
        "",
        "## Controlled development results",
        "",
        "| Model | Best epoch | PCK@0.05 | PCK@0.10 | PCK@0.15 | NME |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, run in development_runs:
        best = best_history(run)
        if best is None:
            continue
        val = best["validation"]
        lines.append(
            f"| {label} | {best['epoch']} | {fmt(val['pck@0.05'])} | {fmt(val['pck@0.1'])} | {fmt(val['pck@0.15'])} | {fmt(val['normalized_mean_error'])} |"
        )

    benchmark_files = (
        ("ViTPose++ Small (pretrained)", "baseline-vitpose-full.json"),
        ("Plain control", "benchmark-plain-full.json"),
        ("Hybrid", "benchmark-hybrid-full.json"),
        ("Hybrid sharp", "benchmark-hybrid-sharp-full.json"),
        ("Hybrid anchor", "benchmark-hybrid-anchor-full.json"),
        ("Hybrid fast", "benchmark-hybrid-fast-full.json"),
        ("SimpleBaseline R18", "benchmark-simple-r18-full.json"),
        ("KineRes plain", "benchmark-kineres-plain-full.json"),
        ("KineRes graph (3 layers)", "benchmark-kineres-graph-full.json"),
        ("KineRes graph (1 layer)", "benchmark-kineres-graph1-full.json"),
    )
    lines += [
        "",
        "## Complete COCO validation person-crop benchmark",
        "",
        "| Model | Parameters | COCO AP | PCK@0.10 | PCK@0.15 | NME | FP32 latency | FP16 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, filename in benchmark_files:
        path = RUNS / filename
        if not path.exists():
            continue
        result = load(path)
        lines.append(
            f"| {label} | {result['model']['parameters']:,} | {fmt(result['coco']['AP'])} | {fmt(result['pck']['pck@0.1'])} | {fmt(result['pck']['pck@0.15'])} | {fmt(result['pck']['normalized_mean_error'])} | {result['throughput']['fp32']['latency_ms']:.2f} ms | {result['throughput']['fp16']['latency_ms']:.2f} ms |"
        )

    confirmation_runs = (
        ("Plain, seed 9019", "confirm-plain-seed9019"),
        ("Hybrid fast, seed 9019", "confirm-fast-seed9019"),
        ("KineRes plain, seed 9019", "confirm-kineres-plain-seed9019"),
        ("KineRes graph, seed 9019", "confirm-kineres-graph-seed9019"),
    )
    available_confirmation = [(label, best_history(run)) for label, run in confirmation_runs]
    available_confirmation = [(label, best) for label, best in available_confirmation if best]
    if available_confirmation:
        lines += [
            "",
            "## Independent-seed confirmation",
            "",
            "| Model | Best epoch | PCK@0.10 | PCK@0.15 | NME |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, best in available_confirmation:
            val = best["validation"]
            lines.append(
                f"| {label} | {best['epoch']} | {fmt(val['pck@0.1'])} | {fmt(val['pck@0.15'])} | {fmt(val['normalized_mean_error'])} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- The directional encoder was rejected because it underperformed the equal-protocol plain encoder.",
        "- Sharp joint-token pooling improved the graph path, while the more elaborate anchored prior did not help and was rejected.",
        "- Increasing residual branch initialization from 0.001 to 0.1 produced the largest controlled KinePose improvement without adding parameters.",
        "- The current compact architecture is not state of the art and is not yet suitable as a clinical pose extractor. The pretrained ViTPose reference remains substantially more accurate.",
        "- Ground-truth person crops isolate landmark quality; a deployed camera pipeline still needs a person detector and temporal tracking.",
        "",
        "## Sources",
        "",
        "- COCO: https://cocodataset.org/",
        "- Simple Baselines for Human Pose Estimation and Tracking: https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html",
        "- ViTPose++: https://arxiv.org/abs/2212.04246",
        "",
    ]
    output = RUNS / "research-report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
