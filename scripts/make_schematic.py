#!/usr/bin/env python3
"""Draw the CentralDogma-DeltaCC architecture schematic (paper Figure 1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 9})
DNA = "#2274A5"; PROT = "#E83151"; SH = "#32936F"; GRY = "#6b6b6b"


def box(ax, x, y, w, h, text, fc, ec=None, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(ax, p, q, c=GRY, style="-|>", lw=1.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=lw, color=c))


fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

# DNA branch (top)
box(ax, 0.1, 4.9, 1.7, 0.7, "WT / mut\n8192 bp window", "#cfe3f2", ec=DNA, tc="black", fs=8)
box(ax, 2.1, 4.9, 1.3, 0.7, "Evo 2\n(StripedHyena)", DNA, fs=8)
box(ax, 3.7, 4.9, 1.4, 0.7, r"$\Delta h_{DNA}$" + "\n(4096)", "#eef5fb", ec=DNA, tc="black", fs=8)
arrow(ax, (1.8, 5.25), (2.1, 5.25), DNA); arrow(ax, (3.4, 5.25), (3.7, 5.25), DNA)

# PROT branch (bottom)
box(ax, 0.1, 0.4, 1.7, 0.7, "WT / mut\nresidue window", "#f7d2d8", ec=PROT, tc="black", fs=8)
box(ax, 2.1, 0.4, 1.3, 0.7, "ESM-2\n(Transformer)", PROT, fs=8)
box(ax, 3.7, 0.4, 1.4, 0.7, r"$\Delta h_{PROT}$" + "\n(1280)", "#fdeef0", ec=PROT, tc="black", fs=8)
arrow(ax, (1.8, 0.75), (2.1, 0.75), PROT); arrow(ax, (3.4, 0.75), (3.7, 0.75), PROT)

# encoders
box(ax, 5.4, 5.15, 1.35, 0.5, r"$E^{DNA}_{shared}$", SH, fs=8)
box(ax, 5.4, 4.5, 1.35, 0.5, r"$E^{DNA}_{priv}$", DNA, fs=8)
box(ax, 5.4, 0.9, 1.35, 0.5, r"$E^{PROT}_{shared}$", SH, fs=8)
box(ax, 5.4, 0.25, 1.35, 0.5, r"$E^{PROT}_{priv}$", PROT, fs=8)
for y in (5.4, 4.75):
    arrow(ax, (5.1, 5.25), (5.4, y + 0.0 if y > 5 else y), DNA)
arrow(ax, (5.1, 5.25), (5.4, 5.4), DNA); arrow(ax, (5.1, 5.25), (5.4, 4.75), DNA)
arrow(ax, (5.1, 0.75), (5.4, 1.15), PROT); arrow(ax, (5.1, 0.75), (5.4, 0.5), PROT)

# shared latent codes + alignment
box(ax, 7.0, 5.15, 1.1, 0.5, r"$z^{DNA}_{s}$", SH, fs=8)
box(ax, 7.0, 0.9, 1.1, 0.5, r"$z^{PROT}_{s}$", SH, fs=8)
box(ax, 7.0, 4.5, 1.1, 0.5, r"$z^{DNA}_{p}$", DNA, fs=8)
box(ax, 7.0, 0.25, 1.1, 0.5, r"$z^{PROT}_{p}$", PROT, fs=8)
arrow(ax, (6.75, 5.4), (7.0, 5.4), SH); arrow(ax, (6.75, 4.75), (7.0, 4.75), DNA)
arrow(ax, (6.75, 1.15), (7.0, 1.15), SH); arrow(ax, (6.75, 0.5), (7.0, 0.5), PROT)
# alignment double-arrow between shared codes
ax.add_patch(FancyArrowPatch((7.55, 5.1), (7.55, 1.45), arrowstyle="<|-|>", mutation_scale=12,
                             lw=1.6, color=SH, linestyle=(0, (4, 2))))
ax.text(7.9, 3.3, "align +\ncontrast\n(BatchTopK)", color=SH, fontsize=8, va="center")

# BatchTopK + decoders -> reconstruction
box(ax, 8.7, 4.6, 1.2, 0.9, r"$D^{DNA}$" + "\n$\\widehat{\\Delta h}_{DNA}$", "#eef5fb", ec=DNA, tc="black", fs=8)
box(ax, 8.7, 0.35, 1.2, 0.9, r"$D^{PROT}$" + "\n$\\widehat{\\Delta h}_{PROT}$", "#fdeef0", ec=PROT, tc="black", fs=8)
arrow(ax, (8.1, 5.0), (8.7, 5.0), DNA); arrow(ax, (8.1, 4.75), (8.7, 4.9), DNA)
arrow(ax, (8.1, 1.15), (8.7, 0.85), PROT); arrow(ax, (8.1, 0.5), (8.7, 0.7), PROT)

ax.text(5.0, 5.9, "Genome model", color=DNA, fontsize=10, ha="center", weight="bold")
ax.text(5.0, 0.02, "Protein model", color=PROT, fontsize=10, ha="center", weight="bold")
fig.savefig("figures/fig_schematic.pdf", bbox_inches="tight")
fig.savefig("figures/fig_schematic.png", dpi=150, bbox_inches="tight")
print("wrote figures/fig_schematic.pdf")
