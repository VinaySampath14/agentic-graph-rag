"""Generate the agent flow architecture diagram → figures/architecture.png"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from pathlib import Path

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.axis("off")

BLUE   = "#DCEEFF"
GREEN  = "#E4F5E1"
ORANGE = "#FFE8B8"
RED    = "#FFD6D6"
GRAY   = "#D9D9D9"

pos = {
    "query_analyser": (0, 6),
    "router":         (0, 5),
    "naive_retr.":    (-2.2, 4),
    "graph_retr.":    (0, 4),
    "comm_retr.":     (2.2, 4),
    "grade_context":  (0, 2.8),
    "rewrite_query":  (-4, 2.8),
    "web_retr.":      (4, 2.8),
    "generator":      (0, 1.7),
    "force_refusal":  (-3.5, 0.7),
    "grade_answer":   (0, 0.7),
    "END_top":        (3.8, 6),
    "END_left":       (-3.5, -0.4),
    "END_bottom":     (0, -0.4),
}

colors = {
    "query_analyser": BLUE,
    "router":         BLUE,
    "naive_retr.":    GREEN,
    "graph_retr.":    GREEN,
    "comm_retr.":     GREEN,
    "grade_context":  ORANGE,
    "rewrite_query":  BLUE,
    "web_retr.":      GREEN,
    "generator":      BLUE,
    "force_refusal":  RED,
    "grade_answer":   ORANGE,
}

box_w, box_h = 1.8, 0.45


def draw_box(name):
    x, y = pos[name]
    rect = FancyBboxPatch(
        (x - box_w / 2, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        edgecolor="black", facecolor=colors[name], linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(x, y, name, ha="center", va="center",
            fontsize=10, family="monospace", fontweight="bold")


def draw_end(name):
    x, y = pos[name]
    ax.add_patch(Circle((x, y), 0.28, edgecolor="black",
                         facecolor=GRAY, linewidth=1.2))
    ax.text(x, y, "END", ha="center", va="center",
            fontsize=9, fontweight="bold")


def arrow(start, end, dashed=False, label=None, rad=0, offset=(0, 0)):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
        color="gray" if dashed else "black",
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
    ))
    if label:
        lx = (start[0] + end[0]) / 2 + offset[0]
        ly = (start[1] + end[1]) / 2 + offset[1]
        ax.text(lx, ly, label, fontsize=9, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


def top(n):   return (pos[n][0], pos[n][1] + box_h / 2)
def bottom(n): return (pos[n][0], pos[n][1] - box_h / 2)
def left(n):  return (pos[n][0] - box_w / 2, pos[n][1])
def right(n): return (pos[n][0] + box_w / 2, pos[n][1])


for n in colors:
    draw_box(n)
for n in ("END_top", "END_left", "END_bottom"):
    draw_end(n)

# Solid arrows
arrow(bottom("query_analyser"), top("router"))
arrow(bottom("router"), top("naive_retr."))
arrow(bottom("router"), top("graph_retr."))
arrow(bottom("router"), top("comm_retr."))
arrow(bottom("naive_retr."), top("grade_context"))
arrow(bottom("graph_retr."), top("grade_context"))
arrow(bottom("comm_retr."), top("grade_context"))
arrow(left("web_retr."), right("grade_context"))
arrow(bottom("grade_context"), top("generator"), label="pass", offset=(0.35, 0))
arrow(bottom("generator"), top("grade_answer"))
arrow(bottom("grade_answer"), (pos["END_bottom"][0], pos["END_bottom"][1] + 0.28))
arrow(bottom("force_refusal"), (pos["END_left"][0], pos["END_left"][1] + 0.28))

# Dashed arrows
arrow(right("query_analyser"), (pos["END_top"][0] - 0.28, pos["END_top"][1]),
      dashed=True, label="OOD", offset=(0, 0.15))
arrow(left("grade_context"), right("rewrite_query"),
      dashed=True, label="fail", offset=(0, 0.18))
arrow(top("rewrite_query"), left("router"),
      dashed=True, label="rewrite", rad=-0.25, offset=(-0.45, 0.1))
arrow(right("grade_context"), left("web_retr."),
      dashed=True, label="loop=3", offset=(0, 0.18))
arrow(left("grade_context"), top("force_refusal"),
      dashed=True, label="exhausted", rad=0.2, offset=(-0.35, -0.05))

ax.set_xlim(-5.2, 5.2)
ax.set_ylim(-1, 6.6)
plt.tight_layout()

out = Path(__file__).parent.parent / "figures" / "architecture.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
