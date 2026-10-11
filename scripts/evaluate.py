import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve

from src.dataset import LungCTDataset, infer_columns, normalize_label
from src.models import build_model
from src.metrics import compute_metrics
from src.utils import load_yaml, ensure_dir, save_json, device_from_cfg

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    device = device_from_cfg(cfg)

    df = pd.read_csv(cfg["data"]["split_csv"])
    _, _, type_col, _, _ = infer_columns(df)
    test_df = df[df["split"]=="test"].copy()
    test_df["_label"] = test_df[type_col].map(normalize_label)

    ds = LungCTDataset(
        test_df, cfg["data"]["images_dir"],
        image_size=int(cfg["data"].get("image_size",224)),
        train=False,
        preprocessing=cfg["data"].get("preprocessing","minmax"),
        lung_center=float(cfg["data"].get("lung_center",-600)),
        lung_width=float(cfg["data"].get("lung_width",1500))
    )
    dl = DataLoader(ds, batch_size=int(cfg["training"].get("batch_size",16)),
                    shuffle=False, num_workers=int(cfg["training"].get("workers",4)))

    model = build_model(cfg["model"]["architecture"], 2, pretrained=False).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    threshold = float(ck.get("threshold",0.5))

    ys, probs, rows = [], [], []
    for b in dl:
        x = b["image"].to(device)
        p = torch.softmax(model(x), dim=1)[:,1].cpu().numpy()
        y = np.asarray(b["label"])
        ys.extend(y.tolist())
        probs.extend(p.tolist())
        for i in range(len(y)):
            rows.append({
                "uuid": b["uuid"][i],
                "slice": b["slice"][i],
                "y_true": int(y[i]),
                "p_fake": float(p[i]),
                "y_pred": int(p[i] >= threshold),
            })

    out = ensure_dir(Path(cfg["output"]["root"])/"results"/Path(args.checkpoint).parent.name)
    pd.DataFrame(rows).to_csv(out/"predictions.csv", index=False)
    m = compute_metrics(ys, probs, threshold)
    save_json(m, out/"metrics.json")

    cm = confusion_matrix(ys, np.asarray(probs)>=threshold, labels=[0,1])
    ConfusionMatrixDisplay(cm, display_labels=["Real","Fake"]).plot(values_format="d")
    plt.tight_layout()
    plt.savefig(out/"confusion_matrix.png", dpi=200)
    plt.close()

    if len(np.unique(ys)) > 1:
        fpr, tpr, _ = roc_curve(ys, probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(out/"roc_curve.png", dpi=200)
        plt.close()

        precision, recall, _ = precision_recall_curve(ys, probs)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(out/"pr_curve.png", dpi=200)
        plt.close()

    print(m)
    print(f"Saved to: {out}")

if __name__ == "__main__":
    main()
