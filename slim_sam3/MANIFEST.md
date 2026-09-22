# SAM3 Task-Aware Pruning — Complete Bundle

## Layout

- `code/` — the full pruning pipeline, readable end-to-end:
  - `README.md` — pipeline walkthrough (rank → allocate/prune → distill → eval)
  - `rank_mlp_channels_full.py`, `prune_mlp_channels_adaptive.py`,
    `finetune_mlp_pruned_full.py`, `load_slim_sam3.py`, quant/VRAM tools
  - `eval/` — mAP / semantic-IoU harness (`eval_detection_ap.py`) + run scripts
  - `baseline_comparison/BASELINE_COMPARISON.md` — **the eval table**
    (mIoU + false-positive stats, all models)
  - `baseline_comparison/panels/` — 20 six-tile visuals (RGB | GT | Slim | GEM |
    MonoGlass3D | RFENet)
  - `results_1206_logs/` — raw eval logs for every number in the table +
    `matches_*.npz` (per-prediction score/TP dumps for re-thresholding)

- `baselines/` — code (+ checkpoints where kept) for every external baseline:
  - `RFENet/` (IJCAI-23; `ckpts/RFENet_GSD_rx101.pth`; our wiring `infer_fullruns.py`)
  - `MonoGlass3D/` (code + our 2D wiring `infer_fullruns_2d.py`; checkpoint NOT
    included — get `mono_g3d_nn.pth` from their BaiduDisk link in readme.md)
  - `GEM/` (MaskDINO+SAM-B; `GEM_Base_Finetune_GSD-S_...pth`; our wiring
    `infer_fullruns.py`; bundled detectron2 source — build needs gcc >= 9)
  - `yolo11_full_runs_v3/` (our on-domain YOLO11n-seg: best.pt + train args +
    pred-cache script)

- `eval_images_1206/` — the exact benchmark set: 1,206 images
  (`<scene>/<frame>/frame.jpg` + `*_bboxes.json`), ordering in `val_list.txt`
  (10% of full_runs_old, seed 22).

## Headline results (1,206 imgs, GT = SAM3 ViT-L teacher, combined glass+window)

| Model | mIoU | pixel precision | FP % painted |
|---|---:|---:|---:|
| Slim-2816 (ours) | 0.874 | 0.961 | 3.9% |
| YOLO11n v3 (ours, on-domain) | 0.526 | 0.715 | 28.5% |
| MonoGlass3D (zero-shot) | 0.386 | 0.542 | 45.8% |
| GEM (zero-shot) | 0.370 | 0.497 | 50.3% |
| RFENet (zero-shot) | 0.244 | 0.292 | 70.8% |

Slim-2816 instance level (conf 0.5): precision 0.921 / recall 0.911 / 1.55 FP/img.
Not included (kept on the cluster, ask/copy if needed): original sam3.pt (3.3 GB),
slim 2816/1752 checkpoints (`sam3/pruning_algorithms/finetune_full32_mlp_*/
student_best.pt` + `mlp_pruned_meta.json`), MonoGlass3D `mono_g3d_nn.pth`.
The slim pruning CONFIGS are small and remain documented in code/README.md.
Generated 2026-08-11.
