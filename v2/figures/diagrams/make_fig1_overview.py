"""Figure 1: study overview with graphics. Two lanes (Study 1: EAV with a literature prior, no
co-registration; Study 2: Urban Appraisal, co-registered), four stages each: data, how the
environment is linked to the EEG, model, what is tested. Replaces the box-only framework figure
and the separate Study 2 architecture figure."""
from pathlib import Path

from drawio_lib import Diagram

A = Path(__file__).resolve().parents[1] / "graphics" / "assets"
d = Diagram("Study overview", 1700, 780)
BLUE, GREEN, PURPLE, ORANGE, INK, MUTED = "#4A7FB5", "#6A9F4B", "#8064A2", "#D9822B", "#1F2328", "#6B7280"
COLS = [(170, "Data"), (590, "Linking environment and EEG"), (1010, "Models"), (1360, "What is tested")]


def label(x, y, w, text, size=13, color=INK, h=40, align="center"):
    return d.text(x, y, w, h, text, size=size, color=color, align=align)


for x, t in COLS:
    label(x, 8, 320, f"<b>{t}</b>", 17, h=30)

# ---------------------------------------------------------------- Study 1 lane
d.panel(14, 44, 1672, 346, "", "neural")
label(24, 54, 140, "<b>Study 1</b><br>EAV<br><span style='font-size:12px'>re-analysis of the<br>original design</span>",
      16, BLUE, h=110)
d.image(178, 70, 92, 92, A / "lab_person.svg")
d.image(276, 64, 98, 104, A / "eeg_head.png")
label(166, 172, 220, "42 people, lab conversation<br>30-channel EEG, 5 emotions", 12, h=40)
d.image(196, 228, 70, 70, A / "satellite.svg")
d.image(272, 214, 112, 112, A / "astana_rgb_grid.png")
label(166, 330, 220, "Astana, 100 cells of 1 km²<br>Sentinel-2 + OpenStreetMap", 12, h=40)
d.box(540, 70, 380, 300, "", "link", fill="#F4EFFA")
label(550, 76, 360, "<b>No shared place or time</b>", 14, PURPLE, h=26)
d.image(566, 104, 170, 150, A / "prior_subsets.png")
label(740, 104, 172, "each epoch gets a grid vector: drawn by its <b>emotion label</b> in PECM (original, training only), or <b>without the label</b> in the proposed model", 12, h=150,
      align="left")
label(556, 262, 350, "literature-informed prior (PECM):<br>environment assigned, not measured.<br>"
      "Support mismatch: one person's EEG vs a 1 km² city cell", 12, h=70)
d.box(960, 96, 330, 250, "", "grey", fill="#FFFFFF")
for i, (lab, col, fill) in enumerate([("EEG tower", BLUE, "#A9C8EA"), ("graph tower", GREEN, "#BFD9AE")]):
    d.box(990 + i * 160, 120, 110, 110, f"<b>{lab}</b>", "neural" if i == 0 else "env", size=13, fill=fill)
d.arrow(pts=[(1100, 160), (1150, 160)], color=INK, width=2)
d.arrow(pts=[(1150, 190), (1100, 190)], color=INK, width=2)
label(970, 236, 310, "bidirectional attention → 5 classes", 12, h=24)
label(970, 262, 310, "+ tangent-space covariance features<br>(EEG representation, this revision)", 12, MUTED, h=44)
label(960, 312, 330, "(Fig. 3)", 12, MUTED, h=24)
for i, name in enumerate(["neutral", "sadness", "anger", "happiness", "calmness"]):
    d.image(1346 + i * 64, 78, 50, 50, A / f"face_{name}.svg")
d.box(1330, 140, 336, 230, "<b>Unseen people</b> (subject-level CV,<br>lockbox of 12 people)<br><br>"
      "<b>Controls</b>: pure noise, matched random<br>(no environmental data), removed at test<br><br>"
      "<b>Diagnostic</b>: test labels in the prior<br>(upper bound on leakage)", "grey", size=12, fill="#FFFFFF")

# ---------------------------------------------------------------- Study 2 lane
d.panel(14, 406, 1672, 360, "", "env")
label(24, 416, 140, "<b>Study 2</b><br>Urban Appraisal<br><span style='font-size:12px'>extension,<br>co-registered</span>",
      16, GREEN, h=110)
d.image(186, 426, 150, 150, A / "viewer.svg")
label(166, 576, 220, "62 people, 56 Berlin street<br>scenes, 64-channel EEG", 12, h=40)
d.image(196, 628, 170, 98, A / "scene_bars.png")
label(166, 724, 220, "scene composition (segmentation)", 12, h=30)
d.box(540, 432, 380, 300, "", "env", fill="#F4F9F0")
label(550, 438, 360, "<b>Measured together</b>", 14, GREEN, h=26)
d.image(600, 474, 120, 120, A / "viewer.svg")
d.arrow(pts=[(726, 534), (780, 534)], color=GREEN, width=2.5)
d.image(786, 490, 110, 64, A / "eeg_traces.png")
label(556, 606, 350, "the environment of every EEG epoch is the<br>scene in view; 9 ratings per scene and person.<br>"
      "Common support: the viewed scene", 12, h=70)
d.box(960, 458, 330, 250, "<b>Scene</b>: composition; image models<br>(SigLIP, CLIP, DINOv2)<br><br><b>EEG</b>: ERPs, Riemannian, EEGNet,<br>"
      "EEG-Conformer, CBraMod<br><br><b>Fusion</b>: stacking; dual tower with<br>token-level cross-attention", "grey", size=12,
      fill="#FFFFFF")
d.box(1330, 458, 336, 250, "<b>Unseen people × unseen scenes</b><br>(crossed 5 × 4 folds)<br><br>"
      "<b>Controls</b>: noise and shuffled scenes,<br>low-level image statistics, shuffled EEG<br><br>"
      "<b>Personal deviation</b> from the consensus<br><br><b>Decoding</b> the scene's greenery from EEG", "grey", size=12, fill="#FFFFFF")

for y, col in ((220, BLUE), (584, GREEN)):
    d.arrow(pts=[(392, y), (536, y)], color=col, width=3)
    d.arrow(pts=[(922, y), (956, y)], color=col, width=3)
    d.arrow(pts=[(1292, y), (1326, y)], color=col, width=3)
d.save("fig1_overview.drawio")
print("wrote fig1_overview.drawio")
