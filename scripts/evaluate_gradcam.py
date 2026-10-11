import sys, argparse, math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import LungCTDataset, infer_columns, normalize_label, dicom_to_array
from src.preprocessing import minmax01, apply_window_hu
from src.models import build_model, gradcam_target_layer
from src.gradcam import GradCAM, resize_cam, overlay_cam, peak_xy
from src.utils import load_yaml, ensure_dir, save_json, device_from_cfg

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
    fake_df = test_df[test_df["_label"]==1].copy()

    ds = LungCTDataset(
        fake_df, cfg["data"]["images_dir"],
        image_size=int(cfg["data"].get("image_size",224)),
        train=False,
        preprocessing=cfg["data"].get("preprocessing","minmax"),
        lung_center=float(cfg["data"].get("lung_center",-600)),
        lung_width=float(cfg["data"].get("lung_width",1500))
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg["model"]["architecture"],2,pretrained=False).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    cam_engine = GradCAM(model, gradcam_target_layer(model, cfg["model"]["architecture"]))
    out = ensure_dir(Path(cfg["output"]["root"])/"results"/Path(args.checkpoint).parent.name/"gradcam")

    rows = []
    for b in dl:
        gx = float(b["x"][0])
        gy = float(b["y"][0])
        if math.isnan(gx) or math.isnan(gy):
            continue

        x = b["image"].to(device)
        cam, logits = cam_engine(x, class_idx=1)
        prob = float(torch.softmax(logits, dim=1)[0,1].cpu())

        hu = dicom_to_array(b["path"][0])
        if cfg["data"].get("preprocessing","minmax").lower() in ("window","lung_window","ct_window"):
            img01 = apply_window_hu(
                hu,
                float(cfg["data"].get("lung_center",-600)),
                float(cfg["data"].get("lung_width",1500))
            )
        else:
            img01 = minmax01(hu)

        h, w = img01.shape
        cam_r = resize_cam(cam, w, h)
        px, py = peak_xy(cam_r)
        dist = float(((px-gx)**2 + (py-gy)**2)**0.5)

        overlay = overlay_cam(img01, cam_r)
        cv2.circle(overlay, (int(round(gx)), int(round(gy))), 6, (255,255,255), 2)
        cv2.circle(overlay, (px,py), 5, (0,0,0), 2)

        fn = f"{b['uuid'][0]}_{b['slice'][0]}.png".replace("/","_")
        cv2.imwrite(str(out/fn), overlay)

        rows.append({
            "uuid": b["uuid"][0],
            "slice": b["slice"][0],
            "p_fake": prob,
            "gt_x": gx, "gt_y": gy,
            "cam_x": px, "cam_y": py,
            "distance": dist,
        })

    cam_engine.close()
    rdf = pd.DataFrame(rows)
    rdf.to_csv(out/"gradcam_localization.csv", index=False)

    if len(rdf):
        d = rdf["distance"].to_numpy()
        summary = {
            "n": int(len(d)),
            "mean_distance": float(np.mean(d)),
            "median_distance": float(np.median(d)),
            "hit_at_10": float(np.mean(d<=10)),
            "hit_at_20": float(np.mean(d<=20)),
            "hit_at_30": float(np.mean(d<=30)),
        }
    else:
        summary = {"n":0}

    save_json(summary, out/"gradcam_summary.json")
    print(summary)
    print(f"Saved to: {out}")

if __name__ == "__main__":
    main()
