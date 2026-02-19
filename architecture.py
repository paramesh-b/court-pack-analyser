import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

boxes = [
    (0.5, 1.5, "User\nUploads PDF", "#4A90D9"),
    (2.5, 1.5, "Document\nParser", "#5BA85A"),
    (4.5, 1.5, "LLM Extraction\nEngine (Groq)", "#E8A838"),
    (6.5, 1.5, "Benchmark\nComparison", "#D95B5B"),
    (8.5, 1.5, "Risk Scorer\n& Audit Log", "#7B68EE"),
]

for x, y, label, color in boxes:
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - 0.8, y - 0.5), 1.6, 1.0,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white", linewidth=2
    ))
    ax.text(x, y, label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + 0.8
    x2 = boxes[i+1][0] - 0.8
    y = boxes[i][1]
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))

ax.text(5, 3.5, "AI Court Pack Analyser — System Architecture",
        ha='center', fontsize=13, fontweight='bold', color='#2C3E50')

ax.add_patch(mpatches.FancyBboxPatch(
    (3.5, 0.2), 3.0, 0.6,
    boxstyle="round,pad=0.1",
    facecolor="#95A5A6", edgecolor="white", linewidth=1
))
ax.text(5, 0.5, "Hire Rates Database (CSV)",
        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax.annotate("", xy=(5, 0.8), xytext=(6.5, 1.0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dashed"))

plt.tight_layout()
plt.savefig("architecture.png", dpi=150, bbox_inches='tight')
print("Architecture diagram saved as architecture.png")