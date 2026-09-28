import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

classes = [
    "CAR",
    "PEDESTRIAN",
    "BARRIER",
    "TRAFFIC CONE",
    "TRUCK",
    "MOTORCYCLE",
    "BUS",
    "BICYCLE",
    "CONSTR. VEHICLE",
    "TRAILER",
    "OTHER (Rare)"
]

totals = [7619, 5015, 2323, 1378, 649, 471, 410, 243, 196, 60, 174]
trainable = [3973, 2892, 1335, 405, 385, 278, 199, 128, 119, 50, 69]
val_counts = [641, 464, 634, 0, 76, 0, 66, 28, 0, 41, 41]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0e1117")
fig.suptitle("AutoBus Stage 4 — Dataset Class Distribution & Detection Feasibility\n(363 Train Samples / 41 Val Samples | 404 Total Keyframes)", 
             fontsize=14, color="#f0f6fc", fontweight="bold", y=0.98)

# Panel 1: Total vs Trainable instances in 45m ROI
y_pos = np.arange(len(classes))
width = 0.38

ax1.set_facecolor("#05070a")
ax1.set_title("1. All Annotations vs Observable/Trainable (ROI & Pts ≥ 3)", color="#58a6ff", fontsize=12, fontweight="bold", pad=10)
rects1 = ax1.barh(y_pos - width/2, totals, width, label="Total Annotations in Dataset", color="#30363d", edgecolor="#484f58")
rects2 = ax1.barh(y_pos + width/2, trainable, width, label="Trainable Instances (In 45m ROI, Pts ≥ 3)", color="#238636", edgecolor="#2ea043")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(classes, color="#f0f6fc", fontsize=10, fontweight="bold")
ax1.invert_yaxis()
ax1.set_xlabel("Number of 3D Bounding Boxes", color="#8b949e", fontsize=10)
ax1.tick_params(colors="#8b949e")
ax1.grid(True, color="#21262d", linestyle=":", alpha=0.6, axis="x")
ax1.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)

# Add value labels
for rect in rects2:
    w = rect.get_width()
    ax1.text(w + 80, rect.get_y() + rect.get_height()/2, f"{int(w):,}", ha="left", va="center", color="#3fb950", fontsize=8, fontweight="bold")

# Panel 2: Split Distribution & Presence in Validation Set
ax2.set_facecolor("#05070a")
ax2.set_title("2. Train vs Validation Split Presence (Detecting 0-Val Classes)", color="#58a6ff", fontsize=12, fontweight="bold", pad=10)
train_counts = [t - v for t, v in zip(totals, val_counts)]

rects_tr = ax2.barh(y_pos, train_counts, width*1.8, label="Train Split (8 Scenes)", color="#1f6feb", edgecolor="#388bfd", alpha=0.85)
rects_va = ax2.barh(y_pos, val_counts, width*1.8, left=train_counts, label="Validation Split (2 Scenes)", color="#d29922", edgecolor="#e3b341", alpha=0.85)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(classes, color="#f0f6fc", fontsize=10, fontweight="bold")
ax2.invert_yaxis()
ax2.set_xlabel("Number of Annotations across Split", color="#8b949e", fontsize=10)
ax2.tick_params(colors="#8b949e")
ax2.grid(True, color="#21262d", linestyle=":", alpha=0.6, axis="x")
ax2.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)

# Highlight 0-val classes with red warning
for idx, v in enumerate(val_counts):
    if v == 0 and totals[idx] > 100:
        ax2.text(totals[idx] + 80, idx, "ZERO in Val!", color="#f85149", fontsize=8, fontweight="bold", va="center")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out_dir = r"visualizations\stage4_task"
art_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\1c0087c6-dac4-4253-a522-0bdbc5b00eb8"
os.makedirs(out_dir, exist_ok=True)

plt.savefig(os.path.join(out_dir, "stage4_class_distribution.png"), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig(os.path.join(art_dir, "stage4_class_distribution.png"), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print("Saved stage4_class_distribution.png successfully.")
