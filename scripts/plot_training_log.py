"""Parse bestmodel_log.txt and generate training curves + summary."""
import re
import csv
import sys
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

LOG_FILE = Path("bestmodel_log.txt")
OUT_DIR = Path("training_analysis")
OUT_DIR.mkdir(exist_ok=True)

# ── Parse log ──
text = LOG_FILE.read_text(encoding="utf-8")

# Data storage (will be reversed to chronological order)
epochs_data = []

# Parse each epoch block
epoch_pattern = re.compile(
    r"Epoch\s+(\d+)/100 Summary.*?"
    r"Train Loss:\s+([\d.]+).*?"
    r"Val Loss:\s+([\d.]+).*?"
    r"Overfit Ratio:\s+([\d.]+)x.*?"
    r"LR:\s+([\d.e+-]+).*?"
    r"Time:\s+([\d.]+)s.*?"
    r"Global Step:\s+(\d+)",
    re.DOTALL,
)

grad_pattern = re.compile(
    r"Epoch\s+(\d+)/100.*?"
    r"Projector grad norm:\s+([\d.]+).*?"
    r"Query grad norm:\s+([\d.]+)",
    re.DOTALL,
)

embed_pattern = re.compile(
    r"Epoch\s+(\d+)/100.*?"
    r"Cosine sim \(2 samples\):\s+([\d.]+).*?"
    r"Mean norm:\s+([\d.]+).*?"
    r"Std dev:\s+([\d.]+)",
    re.DOTALL,
)

# Split by epoch blocks (reverse order in log)
blocks = re.split(r"(?=Epoch\s+\d+/100 Summary)", text)

for block in blocks:
    m = epoch_pattern.search(block)
    if not m:
        continue
    epoch = int(m.group(1))
    entry = {
        "epoch": epoch,
        "train_loss": float(m.group(2)),
        "val_loss": float(m.group(3)),
        "overfit_ratio": float(m.group(4)),
        "lr": float(m.group(5)),
        "time_s": float(m.group(6)),
        "global_step": int(m.group(7)),
    }

    # Grad norms
    gm = re.search(r"Projector grad norm:\s+([\d.]+)", block)
    qm = re.search(r"Query grad norm:\s+([\d.]+)", block)
    entry["proj_grad"] = float(gm.group(1)) if gm else None
    entry["query_grad"] = float(qm.group(1)) if qm else None

    # Embedding diagnostics
    cm = re.search(r"Cosine sim \(2 samples\):\s+([\d.]+)", block)
    nm = re.search(r"Mean norm:\s+([\d.]+)", block)
    sm = re.search(r"Std dev:\s+([\d.]+)", block)
    entry["cosine_sim"] = float(cm.group(1)) if cm else None
    entry["mean_norm"] = float(nm.group(1)) if nm else None
    entry["std_dev"] = float(sm.group(1)) if sm else None

    epochs_data.append(entry)

# Sort chronologically
epochs_data.sort(key=lambda x: x["epoch"])

print(f"Parsed {len(epochs_data)} epochs")

# ── Export CSV ──
csv_path = OUT_DIR / "training_log.csv"
fieldnames = ["epoch", "train_loss", "val_loss", "overfit_ratio", "lr",
              "proj_grad", "query_grad", "time_s", "global_step",
              "cosine_sim", "mean_norm", "std_dev"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for e in epochs_data:
        w.writerow({k: e.get(k, "") for k in fieldnames})
print(f"CSV saved: {csv_path}")

# ── Print summary table ──
print()
print("┌───────┬──────────┬──────────┬────────┬───────────┬───────────┬──────────┐")
print("│ Epoch │  Train   │   Val    │ Ratio  │ Proj Grad │ Query Grad│    LR    │")
print("├───────┼──────────┼──────────┼────────┼───────────┼───────────┼──────────┤")
for e in epochs_data:
    pg = f"{e['proj_grad']:.4f}" if e['proj_grad'] is not None else "  N/A "
    qg = f"{e['query_grad']:.4f}" if e['query_grad'] is not None else "  N/A "
    print(f"│  {e['epoch']:>3}  │ {e['train_loss']:.4f}  │ {e['val_loss']:.4f}  │ {e['overfit_ratio']:.2f}x  │  {pg}  │  {qg}  │ {e['lr']:.2e} │")
print("└───────┴──────────┴──────────┴────────┴───────────┴───────────┴──────────┘")

# ── Plot ──
epochs = [e["epoch"] for e in epochs_data]
train = [e["train_loss"] for e in epochs_data]
val = [e["val_loss"] for e in epochs_data]
ratio = [e["overfit_ratio"] for e in epochs_data]
proj_g = [e["proj_grad"] for e in epochs_data]
query_g = [e["query_grad"] for e in epochs_data]
lrs = [e["lr"] for e in epochs_data]
cos_sims = [(e["epoch"], e["cosine_sim"]) for e in epochs_data if e["cosine_sim"] is not None]
mean_norms = [(e["epoch"], e["mean_norm"]) for e in epochs_data if e["mean_norm"] is not None]

# Color scheme
COLORS = {
    "train": "#2196F3",
    "val": "#F44336",
    "best": "#4CAF50",
    "ratio": "#9C27B0",
    "grad_p": "#FF9800",
    "grad_q": "#00BCD4",
    "lr": "#607D8B",
    "cosine": "#E91E63",
    "norm": "#795548",
}

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("AnyProjector v0.9.4 — Phase 2 Training (50K VietSpeech, RTX PRO 6000 Blackwell)",
             fontsize=14, fontweight="bold", y=0.98)

# ── 1. Loss Curves ──
ax = axes[0, 0]
ax.plot(epochs, train, "-o", color=COLORS["train"], markersize=4, linewidth=2, label="Train Loss")
ax.plot(epochs, val, "-s", color=COLORS["val"], markersize=4, linewidth=2, label="Val Loss")
best_epoch = 15
ax.axvline(x=best_epoch, color=COLORS["best"], linestyle="--", alpha=0.7, linewidth=1.5,
           label=f"Best (E{best_epoch}, val={0.2658:.4f})")
ax.fill_between(epochs, train, val, alpha=0.1, color=COLORS["val"])
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("Train vs Val Loss", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 22.5)

# ── 2. Overfit Ratio ──
ax = axes[0, 1]
colors_ratio = []
for r in ratio:
    if r <= 1.2:
        colors_ratio.append("#4CAF50")  # green
    elif r <= 1.5:
        colors_ratio.append("#FF9800")  # orange
    elif r <= 2.0:
        colors_ratio.append("#FF5722")  # deep orange
    else:
        colors_ratio.append("#F44336")  # red

ax.bar(epochs, ratio, color=colors_ratio, alpha=0.8, edgecolor="white", linewidth=0.5)
ax.axhline(y=1.0, color="#4CAF50", linestyle="--", alpha=0.6, label="Perfect fit (1.0)")
ax.axhline(y=1.2, color="#FF9800", linestyle="--", alpha=0.6, label="OK threshold (1.2)")
ax.axhline(y=1.5, color="#F44336", linestyle="--", alpha=0.6, label="Overfit threshold (1.5)")
ax.axhline(y=3.0, color="#9C27B0", linestyle=":", alpha=0.6, label="Catastrophic (3.0)")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Val/Train Ratio", fontsize=11)
ax.set_title("Overfit Ratio", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")
ax.set_xlim(0.5, 22.5)

# ── 3. Gradient Norms ──
ax = axes[1, 0]
ax.plot(epochs, proj_g, "-d", color=COLORS["grad_p"], markersize=4, linewidth=2, label="Projector Grad")
ax.plot(epochs, query_g, "-^", color=COLORS["grad_q"], markersize=4, linewidth=2, label="Query Grad")
# Highlight the E1 spike
ax.annotate(f"E1: {proj_g[0]:.1f}", xy=(1, proj_g[0]), xytext=(3, proj_g[0]*0.85),
            arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Gradient Norm", fontsize=11)
ax.set_title("Gradient Flow", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 22.5)

# ── 4. Learning Rate ──
ax = axes[1, 1]
ax.plot(epochs, lrs, "-", color=COLORS["lr"], linewidth=2.5)
ax.fill_between(epochs, 0, lrs, alpha=0.15, color=COLORS["lr"])
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Learning Rate", fontsize=11)
ax.set_title("LR Schedule (Warmup + Cosine Decay)", fontsize=12, fontweight="bold")
ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 22.5)

# ── 5. Cosine Similarity (Embedding Quality) ──
ax = axes[2, 0]
if cos_sims:
    cs_epochs, cs_vals = zip(*cos_sims)
    ax.plot(cs_epochs, cs_vals, "-o", color=COLORS["cosine"], markersize=6, linewidth=2, label="Cosine Sim")
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="Collapse zone (>0.95)")
    ax.fill_between(cs_epochs, 0.95, 1.0, alpha=0.1, color="red")
    # Annotate collapse vs differentiation
    for ep, cv in cos_sims:
        if cv > 0.95:
            ax.annotate("⚠️", xy=(ep, cv), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Embedding Differentiation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.88, 1.01)

# ── 6. Embedding Mean Norm ──
ax = axes[2, 1]
if mean_norms:
    mn_epochs, mn_vals = zip(*mean_norms)
    ax.plot(mn_epochs, mn_vals, "-s", color=COLORS["norm"], markersize=6, linewidth=2, label="Mean Norm")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Embedding Norm", fontsize=11)
    ax.set_title("Embedding Scale Growth", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = OUT_DIR / "training_curves_v094_vlsp.png"
fig.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\n📊 Charts saved: {plot_path}")

# ── Key findings ──
print("\n" + "=" * 60)
print("  KEY FINDINGS")
print("=" * 60)
print(f"  Dataset:        50K VietSpeech (VLSP)")
print(f"  Hardware:        RTX PRO 6000 Blackwell (95GB VRAM)")
print(f"  Batch:           90 × 2 = 180 effective")
print(f"  Total Epochs:    22 (early stopped at patience 7/7)")
print(f"  Total Time:      ~10.2 hours")
print(f"  Best Model:      Epoch 15, val_loss = 0.2658")
print(f"  Best Train:      0.1977 (at best epoch)")
print(f"  Best Overfit:    1.34x (at best epoch)")
print(f"  Final Overfit:   4.96x (epoch 22 — severe)")
print(f"")
print(f"  🔑 Observations:")
print(f"  - E1-E2: Embedding collapse (cosine > 0.99)")
print(f"  - E3-E6: Near-collapse resolving (cosine 0.98 → 0.97)")
print(f"  - E8-E15: Healthy learning (ratio 0.88-1.34x)")
print(f"  - E15: BEST checkpoint (val=0.2658, ratio=1.34x)")
print(f"  - E16+: Overfitting accelerates (ratio 1.59 → 4.96x)")
print(f"  - E17-E18: Val improved but overfit > 1.5 → skipped save")
print(f"  - Grad norm stable 0.4-1.0 after E3 (healthy)")
