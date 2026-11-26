"""Figure 1: overview of the two studies, from data to what is tested. Compact canvas so that text and
icons stay legible at the width of the printed page."""
from pathlib import Path

from drawio_lib import Diagram

A = Path(__file__).resolve().parents[1] / "graphics" / "assets"
d = Diagram("Study overview", 944, 626)
BLUE, GREEN, PURPLE, INK, MUTED = "#4A7FB5", "#6A9F4B", "#8064A2", "#1F2328", "#5B6270"
H, T, B, S = 19, 17, 16, 15                                    # header, title, body, small font sizes
COLS = [(18, 210, "Data"), (250, 230, "How they are linked"), (502, 210, "Models"), (734, 200, "What is tested")]
CLS = ["neutral", "sadness", "anger", "happiness", "calmness"]


def text(x, y, w, h, html, size=B, color=INK, align="center"):
    return d.text(x, y, w, h, html, size=size, color=color, align=align)


for x, w, name in COLS:
    text(x, 6, w, 32, f"<b>{name}</b>", H)


def row_arrows(y, color):
    for (x0, w0, _), (x1, _, _) in zip(COLS[:-1], COLS[1:]):
        d.arrow(pts=[(x0 + w0 + 2, y), (x1 - 2, y)], color=color, width=3)


# ======================= Study 1: EAV
d.panel(8, 44, 936, 282, "", "neural")
text(18, 48, 700, 28, "<b>Study 1 · EAV</b>&nbsp;&nbsp;re-analysis of the original design", T, "#2F5E8E", "left")
d.image(22, 82, 62, 62, A / "lab_person.svg")
d.image(88, 82, 62, 62, A / "eeg_head.png")
text(150, 84, 82, 60, "42 people<br>30-ch EEG", S, align="left")
text(18, 146, 210, 26, "5 emotions, in a laboratory", S)
d.image(24, 186, 44, 44, A / "satellite.svg")
d.image(74, 176, 80, 80, A / "astana_rgb_grid.png")
text(156, 178, 74, 78, "Astana<br>100 cells<br>× 4 layers", S, align="left")
text(18, 260, 210, 40, "Sentinel-2, OpenStreetMap", S)
d.box(250, 80, 230, 236, "", "link", fill="#F4EFFA")
text(250, 84, 230, 26, "<b>No shared place or time</b>", B, PURPLE)
d.image(296, 110, 138, 117, A / "prior_subsets.png")
text(256, 230, 218, 80, "each epoch gets a grid vector drawn by its <b>emotion label</b> (PECM, training) "
     "or <b>without the label</b> (proposed)", S)
d.box(502, 80, 210, 236, "", "grey", fill="#FFFFFF")
d.box(514, 96, 84, 58, "EEG<br>tower", "neural", size=B, fill="#A9C8EA")
d.box(616, 96, 84, 58, "graph<br>tower", "env", size=B, fill="#BFD9AE")
d.arrow(pts=[(600, 114), (614, 114)], color=INK, width=2)
d.arrow(pts=[(614, 138), (600, 138)], color=INK, width=2)
text(508, 160, 198, 46, "bidirectional attention<br>→ 5 emotions", B)
text(508, 210, 198, 60, "+ tangent-space covariance features (this revision)", S, MUTED)
text(508, 276, 198, 30, "(Figure 3)", S, MUTED)
d.box(734, 80, 200, 236, "", "grey", fill="#FFFFFF")
for i, name in enumerate(CLS):
    d.image(741 + i * 38, 88, 34, 34, A / f"face_{name}.svg")
text(740, 128, 190, 184, "<b>Primary</b>: trial accuracy vs the EEG tower and vs matched random<br>"
     "<b>Unseen people</b>: CV and a 12-person lockbox<br><b>Controls</b>: noise, matched "
     "random, removed at test<br><b>Equivalence</b>: ±3 points<br><b>Published splits</b>: P1–P5", S, align="left")
row_arrows(198, BLUE)

# ======================= Study 2: Urban Appraisal
d.panel(8, 336, 936, 282, "", "env")
text(18, 340, 700, 28, "<b>Study 2 · Urban Appraisal</b>&nbsp;&nbsp;extension with co-registered data", T, "#4E7A36", "left")
d.image(20, 372, 86, 86, A / "viewer.svg")
text(110, 378, 120, 78, "62 people<br>56 Berlin scenes<br>64-ch EEG", S, align="left")
d.image(34, 466, 150, 87, A / "scene_bars.png")
text(18, 556, 210, 40, "scene composition<br>(segmentation)", S)
d.box(250, 372, 230, 236, "", "env", fill="#F4F9F0")
text(250, 376, 230, 26, "<b>Measured together</b>", B, "#4E7A36")
d.image(268, 406, 76, 76, A / "viewer.svg")
d.arrow(pts=[(348, 444), (374, 444)], color=GREEN, width=3)
d.image(378, 414, 92, 60, A / "eeg_traces.png")
text(256, 490, 218, 112, "the environment of every EEG epoch is the scene in view<br><br>"
     "<b>common support:<br>the viewed scene</b>", S)
d.box(502, 372, 210, 236, "", "grey", fill="#FFFFFF")
text(510, 380, 196, 222, "<b>Scene</b>: composition; image models (SigLIP, CLIP, DINOv2, AIMv2)<br>"
     "<b>EEG</b>: ERPs, Riemannian, EEGNet, EEG-Conformer, CBraMod<br><b>Fusion</b>: stacking; dual tower",
     S, align="left")
d.box(734, 372, 200, 236, "", "grey", fill="#FFFFFF")
text(740, 380, 190, 222, "<b>Unseen people × unseen scenes</b> (crossed 5 × 4 folds)<br><b>Controls</b>: noise, "
     "shuffled scenes, low-level image statistics<br><b>Decoding</b> greenery from EEG<br>"
     "<b>Published benchmark</b> (group means)", S, align="left")
row_arrows(490, GREEN)
d.save("fig1_overview.drawio")
print("wrote fig1_overview.drawio")
