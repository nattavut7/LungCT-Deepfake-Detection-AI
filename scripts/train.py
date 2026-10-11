import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dataset import LungCTDataset, infer_columns, normalize_label
from src.models import build_model
from src.losses import make_loss
from src.metrics import compute_metrics, best_f1_threshold
from src.utils import load_yaml, seed_everything, ensure_dir, save_json, device_from_cfg

@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ys, probs = [], []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        y = torch.as_tensor(b["label"], dtype=torch.long, device=device)
        p = torch.softmax(model(x), dim=1)[:,1]
        ys.extend(y.cpu().numpy().tolist())
        probs.extend(p.cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else int(cfg["project"].get("seed", 2026))
    seed_everything(seed)
    device = device_from_cfg(cfg)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(cfg["data"]["split_csv"])
    _, _, type_col, _, _ = infer_columns(df)
    df["_label"] = df[type_col].map(normalize_label)
    train_df = df[df["split"]=="train"].copy()
    val_df = df[df["split"]=="val"].copy()

    dargs = dict(
        images_dir=cfg["data"]["images_dir"],
        image_size=int(cfg["data"].get("image_size",224)),
        preprocessing=cfg["data"].get("preprocessing","minmax"),
        lung_center=float(cfg["data"].get("lung_center",-600)),
        lung_width=float(cfg["data"].get("lung_width",1500)),
    )
    train_ds = LungCTDataset(train_df, train=True, **dargs)
    val_ds = LungCTDataset(val_df, train=False, **dargs)

    sampler = None
    shuffle = True
    if bool(cfg["imbalance"].get("balanced_sampling", False)):
        counts = train_df["_label"].value_counts().to_dict()
        weights = train_df["_label"].map(lambda y: 1.0/counts[int(y)]).values
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            len(weights),
            replacement=True
        )
        shuffle = False

    bs = int(cfg["training"].get("batch_size",16))
    nw = int(cfg["training"].get("workers",4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)

    model_name = cfg["model"]["architecture"]
    model = build_model(model_name, 2, bool(cfg["model"].get("pretrained",True))).to(device)

    counts = train_df["_label"].value_counts().to_dict()
    n = len(train_df)
    class_weights = [n/(2*max(counts.get(0,1),1)), n/(2*max(counts.get(1,1),1))]
    criterion = make_loss(
        cfg["imbalance"]["loss"],
        class_weights=class_weights,
        gamma=float(cfg["imbalance"].get("focal_gamma",2.0)),
        device=device
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate",1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay",1e-4))
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    amp = bool(cfg["training"].get("amp",True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_name = f'{model_name}_{cfg["imbalance"]["loss"]}_seed{seed}'
    out_dir = ensure_dir(Path(cfg["output"]["root"])/"checkpoints"/run_name)
    log_dir = ensure_dir(Path(cfg["output"]["root"])/"logs")
    history_path = log_dir/f"{run_name}.csv"

    start_epoch, best_pr, bad_epochs = 1, -1.0, 0
    history = []
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck["epoch"])+1
        best_pr = float(ck.get("best_pr_auc",-1.0))
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"Resumed from epoch {ck['epoch']}")

    max_epochs = int(cfg["training"].get("epochs",100))
    early = int(cfg["training"].get("early_stopping",10))

    for epoch in range(start_epoch, max_epochs+1):
        model.train()
        train_losses = []
        for b in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{max_epochs}"):
            x = b["image"].to(device, non_blocking=True)
            y = torch.as_tensor(b["label"], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        yv, pv = infer(model, val_loader, device)
        threshold = best_f1_threshold(yv, pv)
        m = compute_metrics(yv, pv, threshold)
        pr = m["pr_auc"] if m["pr_auc"] is not None else -1.0
        sched.step(pr)

        row = {"epoch":epoch, "train_loss":float(np.mean(train_losses)), **m, "lr":opt.param_groups[0]["lr"]}
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        state = {
            "epoch":epoch,
            "model":model.state_dict(),
            "optimizer":opt.state_dict(),
            "best_pr_auc":best_pr,
            "threshold":threshold,
            "seed":seed,
            "config":cfg
        }
        torch.save(state, out_dir/"last.pt")

        if pr > best_pr:
            best_pr = pr
            state["best_pr_auc"] = best_pr
            torch.save(state, out_dir/"best.pt")
            save_json(m, out_dir/"best_metrics.json")
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_pr_auc={pr:.4f} val_f1={m['f1']:.4f} "
            f"val_recall={m['recall']:.4f} threshold={threshold:.2f}"
        )

        if bad_epochs >= early:
            print("Early stopping.")
            break

    print(f"Best PR-AUC: {best_pr:.4f}")
    print(f"Best checkpoint: {out_dir/'best.pt'}")

if __name__ == "__main__":
    main()
