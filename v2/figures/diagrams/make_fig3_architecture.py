"""Figure 3: the Study 1 dual-tower architecture, drawn with real-data graphics (EAV EEG traces,
feature map and covariance of one epoch; the Sentinel-2/OSM layers and graph of the Astana grid;
the prior's cell subsets) and hand-drawn icons. Replaces the box-only architecture figure."""
from pathlib import Path

from drawio_lib import Diagram

A = Path(__file__).resolve().parents[1] / "graphics" / "assets"
d = Diagram("Dual-tower architecture (graphical)", 1700, 1010)
BLUE, GREEN, PURPLE, ORANGE, INK = "#4A7FB5", "#6A9F4B", "#8064A2", "#D9822B", "#1F2328"
CLS = [("neutral", "#9E9E9E"), ("sadness", "#4A7FB5"), ("anger", "#D6604D"), ("happiness", "#F2C14E"),
       ("calmness", "#6A9F4B")]


def label(x, y, w, text, size=13, color=INK, h=40, align="center"):
    return d.text(x, y, w, h, text, size=size, color=color, align=align)


def vec(x, y, n, color, w=14, h=22):
    """An embedding drawn as a row of cells."""
    for i in range(n):
        d.box(x + i * w, y, w, h, "", "grey", fill=color, stroke_w=0.8, rounded=False)


# ======================= EEG branch
d.panel(14, 14, 600, 760, "EEG branch", "neural", size=19)
d.image(34, 74, 104, 104, A / "lab_person.svg")
label(20, 182, 130, "EAV participant,<br>lab conversation", 12)
d.image(170, 62, 118, 126, A / "eeg_head.png")
label(160, 186, 140, "30 electrodes", 12)
d.image(320, 68, 270, 120, A / "eeg_traces.png")
label(320, 186, 270, "one 5 s epoch (3 s shown)", 12)
d.arrow(pts=[(314, 240), (314, 268)], color=BLUE, width=2.5)

d.image(70, 282, 72, 128, A / "feature_map.png")
label(30, 412, 160, "DE + PSD,<br>30 channels × 10", 12)
d.image(300, 282, 124, 124, A / "cov_matrix.png")
label(250, 412, 230, "band covariances, re-centred<br>per person, tangent space", 12)
d.text(186, 320, 90, 40, "<b>+</b>", size=28, color=BLUE)

# tokens -> transformer -> embedding
d.box(40, 486, 550, 170, "", "neural", fill="#F3F7FC")
label(40, 490, 550, "<b>EEG tower</b>: channel tokens + covariance token → 2-layer transformer encoder", 13, h=30)
for i in range(30):
    d.box(58 + i * 15, 530, 12, 26, "", "grey", fill="#A9C8EA", stroke_w=0.6, rounded=False)
d.box(58 + 30 * 15 + 8, 530, 26, 26, "", "grey", fill="#F6C28B", stroke_w=0.6, rounded=False)
label(58, 556, 460, "30 channel tokens", 11, MUTED := "#6B7280", h=24, align="left")
label(500, 556, 90, "covariance", 11, MUTED, h=24)
for k in range(3):
    d.box(120 + k * 10, 590 - k * 6, 360, 40, "", "grey", fill="#E4EEF8" if k < 2 else "#FFFFFF", stroke_w=0.8)
label(140, 584, 340, "self-attention across channels (×2 layers)", 12, h=28)
d.arrow(pts=[(314, 438), (314, 482)], color=BLUE, width=2.5)
d.arrow(pts=[(314, 658), (314, 690)], color=BLUE, width=2.5)
vec(202, 696, 16, "#4A7FB5")
label(150, 722, 330, "256-D EEG embedding  <i>z</i><sub>EEG</sub>", 13, h=30)

# ======================= Environmental branch
d.panel(1086, 14, 600, 760, "Environmental branch (Astana, 10 × 10 km)", "env", size=19)
d.image(1100, 78, 86, 86, A / "satellite.svg")
d.image(1196, 64, 150, 150, A / "astana_rgb_grid.png")
label(1180, 214, 180, "Sentinel-2: NDVI, NDBI", 12)
d.image(1376, 64, 150, 150, A / "astana_osm.png")
label(1362, 214, 180, "OpenStreetMap:<br>buildings, roads", 12, h=40)
d.arrow(pts=[(1386, 258), (1386, 284)], color=GREEN, width=2.5)
d.image(1150, 290, 104, 104, A / "grid_ndvi.png")
d.image(1270, 290, 104, 104, A / "grid_building.png")
label(1130, 396, 270, "100 cells × 4 indicators<br>(NDVI, NDBI, building share, road density)", 12, h=40)
d.image(1430, 280, 140, 140, A / "grid_graph.png")
label(1405, 420, 190, "graph of neighbouring cells", 12, h=24)
d.box(1110, 486, 552, 170, "", "env", fill="#F4F9F0")
label(1110, 490, 552, "<b>Graph tower</b>: <i>v</i><sub>env</sub> added to every node → 2 GCN layers → GRU (length one) → mean", 13, h=30)
for i in range(10):
    d.box(1150 + i * 46, 540, 30, 30, "", "env", fill="#CFE3C2", stroke_w=0.8)
    if i < 9:
        d.arrow(pts=[(1180 + i * 46, 555), (1196 + i * 46, 555)], color="#8FB57A", width=1, end="none")
label(1150, 580, 470, "message passing between neighbouring cells", 11, MUTED, h=24)
d.arrow(pts=[(1386, 438), (1386, 482)], color=GREEN, width=2.5)
d.arrow(pts=[(1386, 658), (1386, 690)], color=GREEN, width=2.5)
vec(1274, 696, 16, "#6A9F4B")
label(1220, 722, 330, "256-D environmental embedding  <i>z</i><sub>ENV</sub>", 13, h=30)

# ======================= Fusion
d.box(650, 300, 400, 260, "", "grey", fill="#F6F6F6", stroke_w=1.5)
label(650, 306, 400, "<b>Bidirectional cross-modal attention</b>", 15, h=30)
d.box(700, 360, 46, 150, "<i>z</i><sub>EEG</sub>", "neural", size=13, fill="#A9C8EA")
d.box(954, 360, 46, 150, "<i>z</i><sub>ENV</sub>", "env", size=13, fill="#BFD9AE")
d.arrow(pts=[(748, 400), (952, 400)], color=BLUE, width=2.5)
label(760, 372, 180, "EEG queries environment", 11, h=24)
d.arrow(pts=[(952, 470), (748, 470)], color=GREEN, width=2.5)
label(760, 476, 180, "environment queries EEG", 11, h=24)
label(660, 512, 380, "one token per tower: each direction is a learned linear map", 11, MUTED, h=40)
d.arrow(pts=[(612, 707), (698, 435)], via=[(636, 707), (636, 435)], color=BLUE, width=2.5)
d.arrow(pts=[(1088, 707), (1002, 435)], via=[(1064, 707), (1064, 435)], color=GREEN, width=2.5)
d.box(700, 592, 300, 44, "concatenate (512) → MLP → softmax", "out", size=13, fill="#FCEEE2")
d.arrow(pts=[(850, 562), (850, 590)], color=ORANGE, width=2.5)
d.arrow(pts=[(850, 638), (850, 664)], color=ORANGE, width=2.5)
for i, (name, col) in enumerate(CLS):
    x = 664 + i * 76
    d.image(x + 8, 668, 56, 56, A / f"face_{name}.svg")
    label(x - 6, 724, 88, name, 12, h=24)
label(650, 752, 400, "<b>five EAV affective states</b>", 13, h=26)
d.box(690, 196, 320, 66, "alignment loss  ‖<i>z</i><sub>EEG</sub> − <i>z</i><sub>ENV</sub>‖²<br>"
      "(added to cross-entropy, λ = 0.1)", "link", size=12, fill="#F4EFFA", dashed=True)

# ======================= PECM: where the environmental vector comes from
d.panel(14, 800, 1672, 196, "Where each epoch's environmental vector v<sub>env</sub> comes from (PECM, literature-informed prior)",
        "link", size=15)
d.image(40, 838, 186, 156, A / "prior_subsets.png")
d.box(250, 846, 470, 64, "<b>Training</b>: drawn from the grid cells of the epoch's class<br>"
      "(upper-quartile exposure for anger and sadness, upper-quartile NDVI<br>for happiness and calm, interquartile cells for neutral)",
      "link", size=12, fill="#FFFFFF", align="left")
d.box(250, 920, 470, 54, "<b>Validation and test</b>: drawn from all cells, independently of the label,<br>"
      "so at test time <i>v</i><sub>env</sub> carries no information about the class", "link", size=12, fill="#FFFFFF",
      align="left")
d.box(760, 846, 440, 128, "<b>Controls</b> (same model, other inputs)<br>• pure noise, no structure<br>"
      "• matched random: same class means and covariances, no environmental data<br>"
      "• environment removed at test (grid mean)<br>• random pairing, shuffled labels, reversed prior", "link",
      size=12, fill="#FFFFFF", align="left")
d.box(1230, 846, 430, 128, "<b>Diagnostic only</b>: test vectors drawn with the true label<br>"
      "(an upper bound on label leakage; never a deployable result)", "link", size=12, fill="#FFFFFF",
      align="left", dashed=True)
d.arrow(pts=[(1640, 800), (1640, 658)], color=PURPLE, width=2.5)
d.arrow(pts=[(704, 358), (760, 264)], via=[(704, 290), (760, 290)], color=PURPLE, width=1.5, dashed=True, end="none")
d.arrow(pts=[(996, 358), (940, 264)], via=[(996, 290), (940, 290)], color=PURPLE, width=1.5, dashed=True, end="none")
d.save("fig3_architecture.drawio")
print("wrote fig3_architecture.drawio")
