# KinePose: experimental architecture specification

KinePose is being developed from random initialization for adult 2D pose
estimation. It does not reuse a pretrained pose encoder, heatmap head, or
checkpoint. The design is experimental; “new” here means newly implemented
for this project, not a claim of peer-reviewed novelty.

## Core design

1. **Directional spatial encoder**
   Each mixer uses local 3×3, horizontal 1×9, and vertical 9×1 depthwise
   branches. An image-conditioned gate selects their contribution. The intent
   is to capture long narrow limb evidence without paying for global attention
   at every image position.
2. **Gated multi-resolution fusion**
   Four feature scales are fused top-down. Learned gates decide whether a
   location should retain high-resolution edge evidence or accept deeper
   semantic context.
3. **Limb orientation fields**
   The model predicts dense unit-vector fields for 12 skeleton edges. These
   fields are supervised in tubes around visible limbs and are embedded back
   into the joint-token features.
4. **Skeleton graph tokens**
   Coarse heatmaps pool one token per joint. Three recurrent graph blocks pass
   messages only along the human skeleton. Updated tokens are projected back
   against the high-resolution pixel features to refine all 17 heatmaps.
5. **Distributional output**
   Final coordinates combine heatmap expectation and learned sub-pixel offsets.
   A joint-wise variance head is trained for uncertainty calibration.

At width 32, the full model has approximately 2.0 million parameters. At width
48 it has approximately 4.4 million parameters.

## Testable hypotheses

- Graph reasoning should improve wrists and ankles under partial occlusion.
- Limb fields should improve distal-joint PCK and rotation robustness.
- The full architecture should outperform both ablations under the identical
  training budget and seed.
- Width 48 should improve accuracy without exceeding the local real-time
  latency target on the RTX 4060 Ti.

## Benchmark protocol

- Dataset: COCO 2017 train and validation person keypoints.
- Model inputs: identical 1.25× ground-truth person crops at 256×192.
- Architecture controls: a conventional local-CNN heatmap model, directional
  model without graph reasoning, directional model without field guidance, and
  the full model. Each receives 24,000 fixed training crops, 1,800 fixed
  validation crops, eight epochs, and seed 3407.
- A fifth hybrid control applies the limb fields and graph solver to the plain
  encoder. It has exactly the same parameter count as the plain control and
  isolates the value of the kinematic reasoning path.
- External baseline: ViTPose++ Small under the same crop pipeline.
- Metrics: COCO OKS AP/AP50/AP75, torso-normalized PCK@0.05/0.075/0.10/0.15,
  mean normalized error, uncertainty error, parameters, and batch-1 GPU latency.
- Final evaluation: full validation split, checkpoint chosen without looking at
  the final full-split result.

The full architecture advances only if it beats both ablations. A larger
training run advances only if it preserves that improvement under at least
three seeds or a substantially larger fixed split.
