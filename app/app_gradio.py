import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import gradio as gr
import numpy as np
import pydicom
import torch

from src.preprocessing import minmax01, apply_window_hu, to_rgb_uint8, make_transform
from src.models import build_model, gradcam_target_layer
from src.gradcam import GradCAM, resize_cam, overlay_cam
from src.utils import load_yaml, device_from_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = device_from_cfg(cfg)
    model = build_model(cfg["model"]["architecture"],2,pretrained=False).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    threshold = float(ck.get("threshold",0.5))

    cam_engine = GradCAM(model, gradcam_target_layer(model, cfg["model"]["architecture"]))
    tfm = make_transform(int(cfg["data"].get("image_size",224)), train=False)

    def predict(file):
        if file is None:
            return None, None, "Please upload a DICOM file."

        ds = pydicom.dcmread(file)
        arr = ds.pixel_array.astype(np.float32)
        arr = arr*float(getattr(ds,"RescaleSlope",1.0)) + float(getattr(ds,"RescaleIntercept",0.0))

        if cfg["data"].get("preprocessing","minmax").lower() in ("window","lung_window","ct_window"):
            img01 = apply_window_hu(
                arr,
                float(cfg["data"].get("lung_center",-600)),
                float(cfg["data"].get("lung_width",1500))
            )
        else:
            img01 = minmax01(arr)

        rgb = to_rgb_uint8(img01)
        x = tfm(rgb).unsqueeze(0).to(device)

        with torch.enable_grad():
            cam, logits = cam_engine(x, 1)

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        cam_r = resize_cam(cam, img01.shape[1], img01.shape[0])
        overlay = overlay_cam(img01, cam_r)

        original = np.clip(img01*255,0,255).astype(np.uint8)
        label = "FAKE" if probs[1] >= threshold else "REAL"
        text = (
            f"{label} | Real: {probs[0]*100:.2f}% | "
            f"Fake: {probs[1]*100:.2f}% | threshold={threshold:.2f}"
        )
        return original, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), text

    with gr.Blocks(title="Lung CT Deepfake Detection") as demo:
        gr.Markdown("# Lung CT Deepfake Detection\nResearch prototype with Grad-CAM")
        inp = gr.File(label="Upload DICOM (.dcm)", file_types=[".dcm"], type="filepath")
        btn = gr.Button("Analyze")
        result = gr.Textbox(label="Prediction")
        with gr.Row():
            out1 = gr.Image(label="Original CT")
            out2 = gr.Image(label="Grad-CAM")
        btn.click(predict, inputs=inp, outputs=[out1,out2,result])

    demo.launch()

if __name__ == "__main__":
    main()
