from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# reuse data plumbing from the GRU script
from train_gru import SeqStore, bucketed_batches, log

CACHE = Path("seq_cache")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--tag", type=str, default="_tr")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()

class TransformerNet(nn.Module):
    def __init__(self, cards: np.ndarray, d_model: int, layers: int, heads: int,
                 ff: int, dropout: float, max_len: int):
        super().__init__()
        dims = [max(3, min(12, round(1.6 * int(c) ** 0.56))) for c in cards]
        self.embs = nn.ModuleList([nn.Embedding(int(c), d) for c, d in zip(cards, dims)])
        self.proj = nn.Linear(sum(dims), d_model)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.max_len = max_len
        self.emb_drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.attn_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        e = torch.cat([emb(x[:, :, j]) for j, emb in enumerate(self.embs)], dim=-1)
        h = self.proj(e)
        pos_ids = torch.arange(L, device=x.device).clamp_max(self.max_len)
        h = self.emb_drop(h + self.pos(pos_ids)[None])
        pad_mask = (torch.arange(L, device=x.device)[None, :] >= lens[:, None].to(x.device))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask)[:, :, None].float()
        mx = h.masked_fill(pad_mask[:, :, None], float("-inf")).max(dim=1).values
        mn = (h * valid).sum(dim=1) / valid.sum(dim=1)
        score = self.attn_score(h).masked_fill(pad_mask[:, :, None], float("-inf"))
        at = (h * torch.softmax(score, dim=1)).sum(dim=1)
        return self.head(torch.cat([mx, mn, at], dim=-1)).squeeze(-1)

@torch.no_grad()
def predict(model, store: SeqStore, idx: np.ndarray, device: str,
            batch: int = 4096) -> np.ndarray:
    model.eval()
    order = np.argsort(store.lens[idx], kind="stable")
    preds = np.zeros(len(idx), dtype=np.float64)
    for i in range(0, len(idx), batch):
        sel = idx[order[i:i + batch]]
        x, lens = store.batch(sel)
        out = model(x.to(device, non_blocking=True), lens)
        preds[order[i:i + batch]] = torch.sigmoid(out).float().cpu().numpy()
    return preds

def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}  transformer d={args.d_model} L={args.layers} h={args.heads}")
    torch.manual_seed(args.seed)

    cards = np.load(CACHE / "cards.npy")
    train, test = SeqStore("train"), SeqStore("test")
    y = np.load(CACHE / "train_y.npy").astype(np.float32)
    folds = np.load(CACHE / "folds.npy")
    n_train = len(y)

    sub_idx = np.arange(n_train)
    run_folds, epochs = args.folds, args.epochs
    if args.quick:
        rng = np.random.default_rng(0)
        sub_idx = np.sort(rng.choice(n_train, size=300_000, replace=False))
        run_folds, epochs = [0], 2
        log("QUICK mode")

    oof = np.zeros(n_train); oof_mask = np.zeros(n_train, dtype=bool)
    test_pred = np.zeros(len(test.ids)); test_all = np.arange(len(test.ids))

    for fold in run_folds:
        t0 = time.time()
        tr_idx = sub_idx[folds[sub_idx] != fold]
        va_idx = sub_idx[folds[sub_idx] == fold]
        log(f"=== Fold {fold}: train={len(tr_idx)} valid={len(va_idx)} ===")
        model = TransformerNet(cards, args.d_model, args.layers, args.heads,
                               args.ff, args.dropout, args.max_len).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        spe = (len(tr_idx) + args.batch - 1) // args.batch
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=epochs * spe, pct_start=0.1)
        lossf = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(args.seed + fold)
        best_auc, best_state, bad = 0.0, None, 0
        for ep in range(1, epochs + 1):
            model.train(); te = time.time(); tot, nb = 0.0, 0
            for bidx in bucketed_batches(rng, tr_idx, train.lens, args.batch):
                x, lens = train.batch(bidx)
                yy = torch.from_numpy(y[bidx]).to(device, non_blocking=True)
                out = model(x.to(device, non_blocking=True), lens)
                loss = lossf(out, yy)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); tot += loss.item(); nb += 1
            va = predict(model, train, va_idx, device)
            auc = roc_auc_score(y[va_idx], va)
            log(f"fold {fold} ep {ep}: loss={tot/nb:.5f} val_auc={auc:.6f} "
                f"({time.time()-te:.0f}s)")
            if auc > best_auc:
                best_auc, bad = auc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                oof[va_idx] = va; oof_mask[va_idx] = True
            else:
                bad += 1
                if bad >= args.patience:
                    log(f"fold {fold}: early stop ep {ep}"); break
        model.load_state_dict(best_state)
        test_pred += predict(model, test, test_all, device) / len(run_folds)
        log(f"fold {fold}: best AUC = {best_auc:.6f} ({(time.time()-t0)/60:.1f} min)")
        del model, best_state; torch.cuda.empty_cache()

    if oof_mask.any():
        log(f"==== OOF AUC = {roc_auc_score(y[oof_mask], oof[oof_mask]):.6f} ====")
    np.save(f"oof{args.tag}.npy", oof)
    np.save(f"oof{args.tag}_mask.npy", oof_mask)
    np.save(f"test_pred{args.tag}.npy", test_pred)
    log("DONE transformer")

if __name__ == "__main__":
    main()
