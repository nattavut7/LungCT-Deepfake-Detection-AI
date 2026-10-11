from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from torch.utils.data import Dataset
from .preprocessing import minmax01, apply_window_hu, to_rgb_uint8, make_transform

FAKE_TYPES = {"fake","fb","fm","fa","tm","tampered","1"}

def infer_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}
    uuid_col = cols.get("uuid") or cols.get("patient_id") or cols.get("patient")
    slice_col = cols.get("slice") or cols.get("slice_id") or cols.get("instance")
    type_col = cols.get("type") or cols.get("label") or cols.get("class")
    x_col = cols.get("x")
    y_col = cols.get("y")
    if uuid_col is None or slice_col is None:
        raise ValueError("labels.csv must contain uuid/patient and slice columns")
    return uuid_col, slice_col, type_col, x_col, y_col

def normalize_label(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    s = str(v).strip().lower()
    return 1 if s in FAKE_TYPES or "fake" in s or "tamper" in s else 0

def dicom_to_array(path):
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept

class LungCTDataset(Dataset):
    def __init__(self, frame, images_dir, image_size=224, train=False,
                 preprocessing="minmax", lung_center=-600.0, lung_width=1500.0):
        self.df = frame.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = make_transform(image_size, train=train)
        self.preprocessing = preprocessing
        self.lung_center = lung_center
        self.lung_width = lung_width
        self.uuid_col, self.slice_col, self.type_col, self.x_col, self.y_col = infer_columns(self.df)

    def __len__(self):
        return len(self.df)

    def _path(self, row):
        uuid = str(row[self.uuid_col])
        sl = str(row[self.slice_col])
        candidates = [
            self.images_dir / uuid / f"{sl}.dcm",
            self.images_dir / uuid / sl,
        ]
        for p in candidates:
            if p.exists():
                return p
        folder = self.images_dir / uuid
        if folder.exists():
            matches = list(folder.glob(f"*{sl}*.dcm"))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No DICOM found for uuid={uuid}, slice={sl}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._path(row)
        hu = dicom_to_array(path)
        if self.preprocessing.lower() in ("window","lung_window","ct_window"):
            img01 = apply_window_hu(hu, self.lung_center, self.lung_width)
        else:
            img01 = minmax01(hu)
        rgb = to_rgb_uint8(img01)
        image = self.transform(rgb)
        label = normalize_label(row[self.type_col]) if self.type_col else 0
        gx = float(row[self.x_col]) if self.x_col and pd.notna(row[self.x_col]) else np.nan
        gy = float(row[self.y_col]) if self.y_col and pd.notna(row[self.y_col]) else np.nan
        return {
            "image": image,
            "label": label,
            "uuid": str(row[self.uuid_col]),
            "slice": str(row[self.slice_col]),
            "x": gx,
            "y": gy,
            "path": str(path),
        }
