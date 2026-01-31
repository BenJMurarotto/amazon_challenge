



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sentence_transformers import SentenceTransformer

import clip
from PIL import Image

from text_extractor import process_text, extract_numeric_features
from image_extractor import data_loader


NUM_SAMPLES   = 50_000
PRICE_CAP     = None
VERBOSE       = True

BATCH_SIZE    = 64
NUM_EPOCHS    = 30
LEARNING_RATE = 3e-3
WEIGHT_DECAY  = 1e-6

EARLY_PATIENCE = 3
MIN_DELTA      = 1e-4

CLIP_MODEL_NAME = "ViT-B/32"
TEXT_MODEL_NAME = "all-MiniLM-L12-v2"

SEED = 42

# Output files
LOSS_FIG_PATH      = "loss_curves_combined.png"
PRED_FIG_PATH      = "pred_vs_true_combined.png"
ERROR_FIG_PATH     = "error_curves_combined.png"
MODEL_SAVE_PATH    = "combined_price_predictor_model.pth"
PRED_SAVE_PATH     = "test_predictions_combined.npz"


np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load data - verbose flag provides distribution plots - turn off while testing/training to speed up runs
df = data_loader(NUM_SAMPLES, PRICE_CAP, verbose=VERBOSE)

image_paths = df["image_path"].tolist()
y = np.log1p(df["price"].values).astype(np.float32)

if VERBOSE:
    pd.Series(y).plot(kind="hist")
    plt.xlabel("log1p(price)")
    plt.ylabel("Count")
    plt.title("Target Distribution (log1p)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


#image feature extaction
clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=device)
clip_model.eval()

def images_to_clip_tensors(paths, batch_size=32):
    embs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = [clip_preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        img_batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            emb = clip_model.encode_image(img_batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embs.append(emb.cpu())

    return torch.cat(embs, dim=0)  # (N, 512) 

image_embeddings = images_to_clip_tensors(image_paths, batch_size=32).numpy().astype(np.float32)


#text extraction
text_list = df["catalog_content"].apply(process_text).tolist()
text_model = SentenceTransformer(TEXT_MODEL_NAME)
text_embeddings = text_model.encode(text_list, batch_size=64, show_progress_bar=True)
text_embeddings = np.asarray(text_embeddings, dtype=np.float32)  # (N, 384)


#numeric feature extraction
quantity_list = df["catalog_content"].apply(extract_numeric_features)
numeric_raw = np.array([list(x) for x in quantity_list], dtype=np.float32)  # (N, 2)


#indicies 
indices = np.arange(len(df))
idx_temp, idx_test = train_test_split(indices, test_size=0.15, random_state=SEED, shuffle=True)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.176, random_state=SEED, shuffle=True)



#scaling and encoding
value_scaler = StandardScaler()
unit_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

value_scaler.fit(numeric_raw[idx_train, 0].reshape(-1, 1))
unit_encoder.fit(numeric_raw[idx_train, 1].reshape(-1, 1))

values_train = value_scaler.transform(numeric_raw[idx_train, 0].reshape(-1, 1))
values_val   = value_scaler.transform(numeric_raw[idx_val,   0].reshape(-1, 1))
values_test  = value_scaler.transform(numeric_raw[idx_test,  0].reshape(-1, 1))

units_train = unit_encoder.transform(numeric_raw[idx_train, 1].reshape(-1, 1))
units_val   = unit_encoder.transform(numeric_raw[idx_val,   1].reshape(-1, 1))
units_test  = unit_encoder.transform(numeric_raw[idx_test,  1].reshape(-1, 1))

#create initial splits

Xnum_train = np.concatenate([values_train, units_train], axis=1).astype(np.float32)
Xnum_val   = np.concatenate([values_val,   units_val], axis=1).astype(np.float32)
Xnum_test  = np.concatenate([values_test,  units_test], axis=1).astype(np.float32)

Xtxt_train = text_embeddings[idx_train]
Xtxt_val   = text_embeddings[idx_val]
Xtxt_test  = text_embeddings[idx_test]

Ximg_train = image_embeddings[idx_train]
Ximg_val   = image_embeddings[idx_val]
Ximg_test  = image_embeddings[idx_test]

y_train = y[idx_train]
y_val   = y[idx_val]
y_test  = y[idx_test]


#create tensors and load to gpu
Xtxt_train_t = torch.from_numpy(Xtxt_train).to(device)
Xtxt_val_t   = torch.from_numpy(Xtxt_val).to(device)
Xtxt_test_t  = torch.from_numpy(Xtxt_test).to(device)

Xnum_train_t = torch.from_numpy(Xnum_train).to(device)
Xnum_val_t   = torch.from_numpy(Xnum_val).to(device)
Xnum_test_t  = torch.from_numpy(Xnum_test).to(device)

Ximg_train_t = torch.from_numpy(Ximg_train).to(device)
Ximg_val_t   = torch.from_numpy(Ximg_val).to(device)
Ximg_test_t  = torch.from_numpy(Ximg_test).to(device)

y_train_t = torch.from_numpy(y_train).unsqueeze(1).to(device)
y_val_t   = torch.from_numpy(y_val).unsqueeze(1).to(device)
y_test_t  = torch.from_numpy(y_test).unsqueeze(1).to(device)

#dataset and loaders to not brick system
train_ds = TensorDataset(Xtxt_train_t, Xnum_train_t, Ximg_train_t, y_train_t)
val_ds   = TensorDataset(Xtxt_val_t,   Xnum_val_t,   Ximg_val_t,   y_val_t)
test_ds  = TensorDataset(Xtxt_test_t,  Xnum_test_t,  Ximg_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nBatches — Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")


#final price model
class PricePredictor(nn.Module):
    def __init__(self, text_dim, numeric_dim, image_dim, dropout_prob=0.2):
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 128), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.num_branch = nn.Sequential(
            nn.Linear(numeric_dim, 32), nn.ReLU(), nn.Dropout(dropout_prob)
        )
        self.img_branch = nn.Sequential(
            nn.Linear(image_dim, 128), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(128, 64), nn.ReLU()
        )

        self.t_norm = nn.LayerNorm(64)
        self.n_norm = nn.LayerNorm(32)
        self.i_norm = nn.LayerNorm(64)

        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 64, 128), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(64, 1)
        )

    def forward(self, x_text, x_num, x_img):
        t = self.t_norm(self.text_branch(x_text))
        n = self.n_norm(self.num_branch(x_num))
        i = self.i_norm(self.img_branch(x_img))
        x = torch.cat([t, n, i], dim=1)
        return self.fusion(x)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


#train loop
net = PricePredictor(
    text_dim=Xtxt_train_t.shape[1],
    numeric_dim=Xnum_train_t.shape[1],
    image_dim=Ximg_train_t.shape[1],
    dropout_prob=0.2
).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()
early = EarlyStopping(patience=EARLY_PATIENCE, min_delta=MIN_DELTA)

train_loss_hist = []
val_loss_hist   = []
train_rmse_hist = []
val_rmse_hist   = []
train_mae_hist  = []
val_mae_hist    = []

best_val = float("inf")
best_state = None


for epoch in range(NUM_EPOCHS):
    net.train()
    running = 0.0

    for xb_text, xb_num, xb_img, yb in train_loader:
        optimizer.zero_grad()
        pred = net(xb_text, xb_num, xb_img)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb_text.size(0)

    train_loss = running / len(train_loader.dataset)
    train_loss_hist.append(train_loss)

    # validation loss
    net.eval()
    v_running = 0.0
    v_count = 0
    with torch.no_grad():
        for xb_text, xb_num, xb_img, yb in val_loader:
            pred = net(xb_text, xb_num, xb_img)
            loss = criterion(pred, yb)
            bs = xb_text.size(0)
            v_running += loss.item() * bs
            v_count += bs

    val_loss = v_running / v_count
    val_loss_hist.append(val_loss)

    # RMSE/MAE in $ space each epoch (train/val)
    with torch.no_grad():
        train_pred_log = net(Xtxt_train_t, Xnum_train_t, Ximg_train_t)
        val_pred_log   = net(Xtxt_val_t,   Xnum_val_t,   Ximg_val_t)

        train_pred = torch.expm1(train_pred_log)
        val_pred   = torch.expm1(val_pred_log)

        y_train_real = torch.expm1(y_train_t)
        y_val_real   = torch.expm1(y_val_t)

        train_rmse = torch.sqrt(torch.mean((train_pred - y_train_real) ** 2)).item()
        val_rmse   = torch.sqrt(torch.mean((val_pred - y_val_real) ** 2)).item()

        train_mae = torch.mean(torch.abs(train_pred - y_train_real)).item()
        val_mae   = torch.mean(torch.abs(val_pred - y_val_real)).item()

    train_rmse_hist.append(train_rmse)
    val_rmse_hist.append(val_rmse)
    train_mae_hist.append(train_mae)
    val_mae_hist.append(val_mae)

    # save best
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

    if early.step(val_loss):
        print(f"\nEarly stopping at epoch {epoch+1}. Best val loss: {best_val:.4f}\n")
        break


#preds and eval on best model
if best_state is not None:
    net.load_state_dict(best_state)
    net.to(device)
    print("Loaded best model weights.\n")

net.eval()
with torch.no_grad():
    train_pred_log = net(Xtxt_train_t, Xnum_train_t, Ximg_train_t)
    val_pred_log   = net(Xtxt_val_t,   Xnum_val_t,   Ximg_val_t)
    test_pred_log  = net(Xtxt_test_t,  Xnum_test_t,  Ximg_test_t)

train_pred = torch.expm1(train_pred_log)
val_pred   = torch.expm1(val_pred_log)
test_pred  = torch.expm1(test_pred_log)

y_train_real = torch.expm1(y_train_t)
y_val_real   = torch.expm1(y_val_t)
y_test_real  = torch.expm1(y_test_t)

train_mse = torch.mean((train_pred - y_train_real) ** 2).item()
val_mse   = torch.mean((val_pred - y_val_real) ** 2).item()
test_mse  = torch.mean((test_pred - y_test_real) ** 2).item()

train_rmse = float(np.sqrt(train_mse))
val_rmse   = float(np.sqrt(val_mse))
test_rmse  = float(np.sqrt(test_mse))

train_mae = torch.mean(torch.abs(train_pred - y_train_real)).item()
val_mae   = torch.mean(torch.abs(val_pred - y_val_real)).item()
test_mae  = torch.mean(torch.abs(test_pred - y_test_real)).item()

print("="*60)
print("Final Metrics ($ space)")
print("="*60)
print(f"Train - MSE: {train_mse:.4f} | RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f}")
print(f"Val   - MSE: {val_mse:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")
print(f"Test  - MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")
print("="*60 + "\n")


#loss curves (log space)
plt.figure(figsize=(8, 5))
plt.plot(train_loss_hist, label="Train MSE (log1p)", marker="o", markersize=3)
plt.plot(val_loss_hist, label="Val MSE (log1p)", marker="x", markersize=4)
plt.axhline(best_val, linestyle="--", alpha=0.6, label=f"Best Val: {best_val:.4f}")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (log1p)")
plt.title("Training & Validation Loss (Combined)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(LOSS_FIG_PATH, dpi=300, bbox_inches="tight")
plt.show()

# predicted vs true (test, $ space)
test_actual = y_test_real.detach().cpu().numpy().flatten()
test_predicted = test_pred.detach().cpu().numpy().flatten()

plt.figure(figsize=(7, 6))
plt.scatter(test_actual, test_predicted, alpha=0.5, edgecolor="black", s=40)
mn = min(test_actual.min(), test_predicted.min())
mx = max(test_actual.max(), test_predicted.max())
plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
plt.xlabel("True Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Predicted vs True (Test) — Combined")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PRED_FIG_PATH, dpi=300, bbox_inches="tight")
plt.show()

# error curves ($ space)
plt.figure(figsize=(8, 5))
plt.plot(train_rmse_hist, label="Train RMSE ($)", marker="o", markersize=3)
plt.plot(val_rmse_hist, label="Val RMSE ($)", marker="x", markersize=4)
plt.plot(train_mae_hist, label="Train MAE ($)", marker="o", markersize=3)
plt.plot(val_mae_hist, label="Val MAE ($)", marker="x", markersize=4)
plt.xlabel("Epoch")
plt.ylabel("Error ($)")
plt.title("RMSE & MAE over Training (Combined)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ERROR_FIG_PATH, dpi=300, bbox_inches="tight")
plt.show()



torch.save(
    {
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_history": train_loss_hist,
        "val_loss_history": val_loss_hist,
        "best_val_loss": best_val,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "config": {
            "NUM_SAMPLES": NUM_SAMPLES,
            "PRICE_CAP": PRICE_CAP,
            "BATCH_SIZE": BATCH_SIZE,
            "NUM_EPOCHS": NUM_EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "EARLY_PATIENCE": EARLY_PATIENCE,
            "MIN_DELTA": MIN_DELTA,
            "CLIP_MODEL_NAME": CLIP_MODEL_NAME,
            "TEXT_MODEL_NAME": TEXT_MODEL_NAME,
            "SEED": SEED,
        }
    },
    MODEL_SAVE_PATH
)
print(f"Saved model to: {MODEL_SAVE_PATH}")

np.savez(
    PRED_SAVE_PATH,
    idx_test=idx_test,
    test_predicted=test_predicted,
    test_actual=test_actual,
    test_pred_log=test_pred_log.detach().cpu().numpy().flatten(),
    test_actual_log=y_test_t.detach().cpu().numpy().flatten(),
)
print(f"Saved predictions to: {PRED_SAVE_PATH}")
