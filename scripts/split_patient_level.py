import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset import infer_columns, normalize_label
from src.utils import load_yaml, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    df = pd.read_csv(cfg["data"]["labels_csv"])
    uuid_col, _, type_col, _, _ = infer_columns(df)
    df["_label"] = df[type_col].map(normalize_label) if type_col else 0
    patient = df.groupby(uuid_col)["_label"].max().reset_index()

    seed = int(cfg["project"].get("split_seed", 2026))
    tr = float(cfg["data"].get("train_ratio", 0.70))
    va = float(cfg["data"].get("val_ratio", 0.15))
    te = float(cfg["data"].get("test_ratio", 0.15))

    p_train, p_tmp = train_test_split(
        patient, test_size=1-tr, random_state=seed,
        stratify=patient["_label"] if patient["_label"].nunique() > 1 else None
    )
    rel_test = te/(va+te)
    p_val, p_test = train_test_split(
        p_tmp, test_size=rel_test, random_state=seed,
        stratify=p_tmp["_label"] if p_tmp["_label"].nunique() > 1 else None
    )

    split_map = {str(x): "train" for x in p_train[uuid_col]}
    split_map.update({str(x): "val" for x in p_val[uuid_col]})
    split_map.update({str(x): "test" for x in p_test[uuid_col]})
    df["split"] = df[uuid_col].astype(str).map(split_map)

    out = Path(cfg["data"].get("split_csv", "outputs/splits/patient_split.csv"))
    ensure_dir(out.parent)
    df.drop(columns=["_label"]).to_csv(out, index=False)

    print(df["split"].value_counts())
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
