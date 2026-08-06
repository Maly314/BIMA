# Adult pose architecture research

This folder is an isolated, reproducible adult 2D pose-estimation study. It
does not change the movement-capture website until a model passes the research
gate, and it does not make a cerebral-palsy or clinical-diagnosis claim.

## Data and protocol

- Dataset: COCO 2017 person keypoints (118,287 train images, 5,000 validation
  images; 129,762/5,430 usable person crops after requiring six keypoints).
- Input: a 1.25x ground-truth person crop resized to 256x192.
- Output: 17 COCO joints on 64x48 heatmaps.
- Controlled development split: 24,000 fixed train crops and 1,800 fixed
  validation crops, seed 3407, eight epochs, identical augmentation, optimizer,
  schedule, losses, and decoding.
- Final evaluation: the complete COCO validation person-crop set with COCO OKS
  AP, torso-normalized PCK, normalized mean error, parameters, and GPU latency.

## Models

- `KinePose`: a new project-specific architecture trained from random
  initialization. See `KINEPOSE_ARCHITECTURE.md`.
- `plain`: equal-code-path local-CNN control.
- `no_graph`, `no_fields`, and `full`: component ablations.
- `hybrid`: the plain encoder plus the new field/graph refinement path. This has
  exactly the same allocated parameter count as `plain`, making it the key
  capacity-controlled comparison.
- `hybrid_sharp`: corrected joint-local token pooling at temperature 0.18. It
  is kept separate from `hybrid` so the temperature-1 design remains an honest
  archived ablation.
- `hybrid_anchor`: samples image/field features at each coarse joint, embeds
  joint coordinates, performs skeleton message passing, predicts bounded joint
  corrections, and renders them back as differentiable kinematic priors.
- `hybrid_fast`: the sharp-token graph design with the exact same parameter
  count, but residual branch scales initialized to 0.1 instead of 0.001.
- `SimpleBaselineR18`: established ResNet-18 plus three-deconvolution top-down
  control, also trained from random initialization with identical supervision.
- `usyd-community/vitpose-plus-small`: external pretrained reference evaluated
  through the identical crop and metric pipeline.

## Commands

```powershell
python -m unittest -v test_kinepose.py
powershell -ExecutionPolicy Bypass -File run_ablations.ps1
python train_kinepose.py --variant hybrid --run-name ablation-hybrid
python train_simplebaseline.py --run-name baseline-simple-r18
python summarize_ablations.py
python benchmark_adult_pose.py --model vitpose --output runs/baseline-vitpose-full.json
python benchmark_adult_pose.py --model kinepose --checkpoint runs/ablation-hybrid/best.pt
python benchmark_adult_pose.py --model simplebaseline --checkpoint runs/baseline-simple-r18/best.pt
```

Datasets, logs, and checkpoints remain local and are ignored by Git. A novel
component advances only when it beats the equal-parameter control and repeats
under another seed; otherwise it is revised or rejected rather than scaled.
