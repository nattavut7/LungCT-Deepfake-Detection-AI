# Lung CT Deepfake Detection under Class Imbalance

GitHub-ready research code for:
- patient-independent splitting
- ResNet18 / DenseNet121 / EfficientNet-B0
- Cross-Entropy / Weighted Cross-Entropy / Focal Loss
- optional Balanced Sampling
- 3-seed experiments
- PR-AUC based model selection
- ROC / PR curves / confusion matrix
- Grad-CAM + localization metrics
- Gradio research prototype

## Recommended Windows location

Extract to:

`F:\Python\LungCT_Deepfake_Detection_TCI1`

Expected dataset:

```text
F:\Datasets\Medical-Deepfakes-Lung-Cancer\
├── labels.csv
└── images\
    └── <UUID>\
        └── <slice>.dcm
```

Edit the paths in `configs\efficientnet_b0_focal.yaml`.

## Install

```bat
python -m pip install -r requirements.txt
```

## Split by patient

```bat
python scripts\split_patient_level.py --config configs\efficientnet_b0_focal.yaml
```

## Train

```bat
python scripts\train.py --config configs\efficientnet_b0_focal.yaml --seed 2026
```

Resume training:

```bat
python scripts\train.py --config configs\efficientnet_b0_focal.yaml --seed 2026 --resume outputs\checkpoints\efficientnet_b0_focal_seed2026\last.pt
```

## Three seeds

```bat
python scripts\run_seeds.py --config configs\efficientnet_b0_focal.yaml --seeds 2026 2027 2028
```

## Evaluate

```bat
python scripts\evaluate.py --config configs\efficientnet_b0_focal.yaml --checkpoint outputs\checkpoints\efficientnet_b0_focal_seed2026\best.pt
```

## Grad-CAM quantitative evaluation

```bat
python scripts\evaluate_gradcam.py --config configs\efficientnet_b0_focal.yaml --checkpoint outputs\checkpoints\efficientnet_b0_focal_seed2026\best.pt
```

## GUI

```bat
python app\app_gradio.py --config configs\efficientnet_b0_focal.yaml --checkpoint outputs\checkpoints\efficientnet_b0_focal_seed2026\best.pt
```

## Publication note

Replace all illustrative manuscript values with results exported by this code before submission.
