from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

CACHE = Path("seq_cache")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--emb-dropout", type=float, default=0.1)
    p.add_argument("--head-dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--attn", action="store_true",
                   help="Add attention pooling alongside max+mean.")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Vectorized padded-batch gathering from the flat row store
class SeqStore:
    def __init__(self, split: str):
        self.feats = np.load(CACHE / f"{split}_feats.npy")
        self.offsets = np.load(CACHE / f"{split}_offsets.npy")
        self.ids = np.load(CACHE / f"{split}_ids.npy")
        self.lens = np.diff(self.offsets).astype(np.int32)

    def batch(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        lens = self.lens[idx]
        L = int(lens.max())
        starts = self.offsets[idx]
        pos = np.arange(L, dtype=np.int64)[None, :]
        flat = starts[:, None] + pos
        mask = pos < lens[:, None]
        flat = np.where(mask, flat, 0)
        x = self.feats[flat]
        x = x * mask[:, :, None]
        return (torch.from_numpy(x.astype(np.int64)),
                torch.from_numpy(lens.astype(np.int64)))


def bucketed_batches(rng: np.random.Generator, idx: np.ndarray, lens: np.ndarray,
                     batch: int, bucket_mult: int = 50) -> list[np.ndarray]:
    # Sort by length
    perm = rng.permutation(idx)
    bucket = batch * bucket_mult
    out: list[np.ndarray] = []
    for i in range(0, len(perm), bucket):
        chunk = perm[i:i + bucket]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        out.extend(chunk[j:j + batch] for j in range(0, len(chunk), batch))
    order = rng.permutation(len(out))
    return [out[i] for i in order]

# Model
class GRUNet(nn.Module):
    def __init__(self, cards: np.ndarray, hidden: int, layers: int,
                 emb_dropout: float, head_dropout: float, attn: bool = False):
        super().__init__()
        dims = [max(3, min(12, round(1.6 * int(c) ** 0.56))) for c in cards]
        self.embs = nn.ModuleList(
            [nn.Embedding(int(c), d) for c, d in zip(cards, dims)]
        )
        in_dim = sum(dims)
        self.emb_drop = nn.Dropout1d(emb_dropout)
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=0.1 if layers > 1 else 0.0)
        out_dim = hidden * 2
        self.attn_score = nn.Linear(out_dim, 1) if attn else None
        feat_dim = out_dim * (3 if attn else 2)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 59) int64
        e = torch.cat([emb(x[:, :, j]) for j, emb in enumerate(self.embs)], dim=-1)
        e = self.emb_drop(e.transpose(1, 2)).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(out.size(1), device=out.device)[None, :]
                < lens[:, None].to(out.device))
        m = mask[:, :, None].float()
        mx = out.masked_fill(~mask[:, :, None], float("-inf")).max(dim=1).values
        mn = (out * m).sum(dim=1) / m.sum(dim=1)
        pooled = [mx, mn]
        if self.attn_score is not None:
            score = self.attn_score(out).masked_fill(~mask[:, :, None], float("-inf"))
            w = torch.softmax(score, dim=1)
            pooled.append((out * w).sum(dim=1))
        return self.head(torch.cat(pooled, dim=-1)).squeeze(-1)

# Train and predict
@torch.no_grad()
def predict(model: nn.Module, store: SeqStore, idx: np.ndarray,
            device: str, batch: int = 4096) -> np.ndarray:
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
    log(f"device={device}  torch={torch.__version__}  "
        f"gpu={torch.cuda.get_device_name(0) if device == 'cuda' else '-'}")
    torch.manual_seed(args.seed)

    cards = np.load(CACHE / "cards.npy")
    train = SeqStore("train")
    test = SeqStore("test")
    y = np.load(CACHE / "train_y.npy").astype(np.float32)
    folds = np.load(CACHE / "folds.npy")

    n_train = len(y)
    sub_idx = np.arange(n_train)
    run_folds = args.folds
    epochs = args.epochs
    if args.quick:
        rng = np.random.default_rng(0)
        sub_idx = np.sort(rng.choice(n_train, size=300_000, replace=False))
        run_folds, epochs = [0], 2
        log("QUICK mode: 300k ids, fold 0, 2 epochs")

    oof = np.zeros(n_train, dtype=np.float64)
    oof_mask = np.zeros(n_train, dtype=bool)
    test_pred = np.zeros(len(test.ids), dtype=np.float64)
    test_idx_all = np.arange(len(test.ids))

    for fold in run_folds:
        t0 = time.time()
        tr_idx = sub_idx[folds[sub_idx] != fold]
        va_idx = sub_idx[folds[sub_idx] == fold]
        log(f"=== Fold {fold}: train={len(tr_idx)} valid={len(va_idx)} ===")

        model = GRUNet(cards, args.hidden, args.layers,
                       args.emb_dropout, args.head_dropout, args.attn).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        steps_per_epoch = (len(tr_idx) + args.batch - 1) // args.batch
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=epochs * steps_per_epoch)
        lossf = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(args.seed + fold)

        best_auc, best_state, bad = 0.0, None, 0
        for ep in range(1, epochs + 1):
            model.train()
            te = time.time()
            tot_loss, nb = 0.0, 0
            for bidx in bucketed_batches(rng, tr_idx, train.lens, args.batch):
                x, lens = train.batch(bidx)
                yy = torch.from_numpy(y[bidx]).to(device, non_blocking=True)
                out = model(x.to(device, non_blocking=True), lens)
                loss = lossf(out, yy)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot_loss += loss.item()
                nb += 1
            va_pred = predict(model, train, va_idx, device)
            auc = roc_auc_score(y[va_idx], va_pred)
            log(f"fold {fold} ep {ep}: loss={tot_loss/nb:.5f} val_auc={auc:.6f} "
                f"({time.time()-te:.0f}s)")
            if auc > best_auc:
                best_auc, bad = auc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                oof[va_idx] = va_pred
                oof_mask[va_idx] = True
            else:
                bad += 1
                if bad >= args.patience:
                    log(f"fold {fold}: early stop at ep {ep}")
                    break

        model.load_state_dict(best_state)
        torch.save(best_state, CACHE / f"gru{args.tag}_fold{fold}.pt")
        test_pred += predict(model, test, test_idx_all, device) / len(run_folds)
        log(f"fold {fold}: best val AUC = {best_auc:.6f}  "
            f"({(time.time()-t0)/60:.1f} min)")
        del model, best_state
        torch.cuda.empty_cache()

    if oof_mask.any():
        log(f"==== OOF AUC (covered {oof_mask.sum()} ids) = "
            f"{roc_auc_score(y[oof_mask], oof[oof_mask]):.6f} ====")
    np.save(f"oof_gru{args.tag}.npy", oof)
    np.save(f"oof_gru{args.tag}_mask.npy", oof_mask)
    np.save(f"test_pred_gru{args.tag}.npy", test_pred)
    log("DONE train_gru")

if __name__ == "__main__":
    main()
