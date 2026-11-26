"""Figure 3: the Study 1 dual-tower architecture, drawn with real-data graphics (EAV EEG traces,
feature map and covariance of one epoch; the Sentinel-2/OSM layers and graph of the Astana grid;
the prior's cell subsets) and icons. Compact canvas so that text and graphics stay legible at the
width of the printed page."""
from pathlib import Path

from drawio_lib import Diagram

A = Path(__file__).resolve().parents[1] / "graphics" / "assets"
d = Diagram("Dual-tower architecture (graphical)", 940, 880)
BLUE, GREEN, PURPLE, ORANGE, INK, MUTED = "#4A7FB5", "#6A9F4B", "#8064A2", "#D9822B", "#1F2328", "#5B6270"
T, L, S, XS = 20, 18, 16, 15                                   # title, label, small, extra-small font sizes
CLS = [("neutral", "neutral"), ("sadness", "sad"), ("anger", "angry"), ("happiness", "happy"), ("calmness", "calm")]


def label(x, y, w, text, size=S, color=INK, h=26, align="center"):
    return d.text(x, y, w, h, text, size=size, color=color, align=align)


def vec(cx, y, n, color, w=12, h=18):
    """An embedding drawn as a row of cells centred on cx."""
    x0 = cx - n * w / 2
    for i in range(n):
        d.box(x0 + i * w, y, w, h, "", "grey", fill=color, stroke_w=0.8, rounded=False)


# ======================= EEG branch
d.panel(8, 8, 316, 560, "EEG branch", "neural", size=T)
d.image(18, 52, 68, 68, A / "lab_person.svg")
label(6, 120, 92, "participant", S)
d.image(100, 48, 76, 76, A / "eeg_head.png")
label(88, 120, 100, "30<br>electrodes", S, h=40)
d.image(188, 52, 126, 83, A / "eeg_traces.png")
label(188, 136, 126, "5 s epoch", S)
d.arrow(pts=[(166, 166), (166, 190)], color=BLUE, width=3)
d.image(40, 196, 58, 96, A / "feature_map.png")
label(8, 294, 124, "DE + PSD<br>(30 × 10)", S, h=40)
d.text(108, 224, 40, 40, "<b>+</b>", size=30, color=BLUE)
d.image(160, 196, 96, 96, A / "cov_matrix.png")
label(130, 294, 160, "band covariances,<br>tangent space", S, h=40)
d.arrow(pts=[(166, 338), (166, 362)], color=BLUE, width=3)
d.box(20, 366, 292, 130, "", "neural", fill="#F3F7FC")
label(20, 370, 292, "<b>EEG tower</b>", L)
for i in range(30):
    d.box(52 + i * 7, 402, 6, 22, "", "grey", fill="#A9C8EA", stroke_w=0.5, rounded=False)
d.box(52 + 30 * 7 + 8, 402, 16, 22, "", "grey", fill="#F6C28B", stroke_w=0.5, rounded=False)
label(20, 426, 292, "30 channel tokens + covariance token", XS, MUTED, h=22)
d.box(74, 452, 184, 32, "2-layer transformer", "neural", size=S, fill="#E4EEF8")
d.arrow(pts=[(166, 498), (166, 514)], color=BLUE, width=3)
vec(166, 518, 12, BLUE)
label(20, 538, 292, "EEG embedding <i>z</i><sub>EEG</sub>", 17)

# ======================= Environmental branch
d.panel(616, 8, 316, 560, "Environmental branch (Astana)", "env", size=T)
d.image(622, 58, 52, 52, A / "satellite.svg")
d.image(678, 48, 110, 110, A / "astana_rgb_grid.png")
label(674, 158, 118, "Sentinel-2", S)
d.image(800, 48, 110, 110, A / "astana_osm.png")
label(790, 158, 132, "OpenStreetMap", S)
d.arrow(pts=[(774, 186), (774, 206)], color=GREEN, width=3)
d.image(628, 210, 86, 86, A / "grid_ndvi.png")
d.image(720, 210, 86, 86, A / "grid_building.png")
label(622, 298, 190, "100 cells × 4 layers", S)
d.image(820, 204, 100, 100, A / "grid_graph.png")
label(810, 304, 120, "cell graph", S)
d.arrow(pts=[(774, 338), (774, 362)], color=GREEN, width=3)
d.box(628, 366, 292, 130, "", "env", fill="#F4F9F0")
label(628, 370, 292, "<b>Graph tower</b>", L)
for i in range(8):
    d.box(638 + i * 36, 402, 24, 24, "", "env", fill="#CFE3C2", stroke_w=1)
    if i < 7:
        d.arrow(pts=[(662 + i * 36, 414), (674 + i * 36, 414)], color="#8FB57A", width=1.5, end="none")
label(628, 428, 292, "<i>v</i><sub>env</sub> added to every node", XS, MUTED, h=22)
d.box(682, 452, 184, 32, "2 GCN layers + GRU", "env", size=S, fill="#E6F1DE")
d.arrow(pts=[(774, 498), (774, 514)], color=GREEN, width=3)
vec(774, 518, 12, GREEN)
label(640, 538, 268, "environmental embedding <i>z</i><sub>ENV</sub>", 17)

# ======================= Fusion (centre)
d.box(342, 22, 250, 72, "alignment loss<br>‖<i>z</i><sub>EEG</sub> − <i>z</i><sub>ENV</sub>‖², λ = 0.1", "link",
      size=S, fill="#F4EFFA", dashed=True)
d.arrow(pts=[(467, 96), (467, 120)], color=PURPLE, width=2, dashed=True)
d.box(332, 124, 270, 226, "", "grey", fill="#F6F6F6", stroke_w=1.5)
label(332, 128, 270, "<b>Bidirectional<br>cross-modal attention</b>", L, h=50)
d.box(348, 188, 46, 140, "<i>z</i><sub>EEG</sub>", "neural", size=S, fill="#A9C8EA")
d.box(540, 188, 46, 140, "<i>z</i><sub>ENV</sub>", "env", size=S, fill="#BFD9AE")
label(396, 198, 142, "EEG → environment", XS)
d.arrow(pts=[(396, 226), (538, 226)], color=BLUE, width=3)
d.arrow(pts=[(538, 290), (396, 290)], color=GREEN, width=3)
label(396, 296, 142, "environment → EEG", XS)
d.arrow(pts=[(240, 527), (348, 258)], via=[(328, 527), (328, 258)], color=BLUE, width=3)
d.arrow(pts=[(700, 527), (586, 258)], via=[(609, 527), (609, 258)], color=GREEN, width=3)
d.arrow(pts=[(467, 352), (467, 370)], color=ORANGE, width=3)
d.box(342, 372, 250, 46, "concatenate → MLP → softmax", "out", size=S, fill="#FCEEE2")
d.arrow(pts=[(467, 420), (467, 436)], color=ORANGE, width=3)
for i, (name, short) in enumerate(CLS):
    x = 336 + i * 54
    d.image(x + 2, 440, 46, 46, A / f"face_{name}.svg")
    label(x - 2, 488, 54, short, XS, h=22)
label(332, 512, 270, "<b>five affective states</b>", S)

# ======================= Where the environmental vector comes from
d.panel(8, 580, 924, 292, "Where each epoch's environmental vector <i>v</i><sub>env</sub> comes from", "link", size=T)
d.image(16, 618, 238, 202, A / "prior_subsets.png")
label(16, 820, 238, "the prior's cell subsets<br>(exposure score vs NDVI)", XS, MUTED, h=40)
d.box(266, 618, 322, 120, "<b>PECM (original design)</b>, training only: drawn from the cells of the epoch's class "
      "(upper-quartile exposure for anger and sadness; upper-quartile NDVI for happiness and calm; "
      "interquartile for neutral)", "link", size=XS, fill="#FFFFFF", align="left")
d.box(266, 748, 322, 112, "<b>Proposed model</b>: drawn from all cells without the label, in training and at "
      "test. At test, both designs draw without the label.", "link", size=XS, fill="#FFFFFF", align="left")
d.box(600, 618, 322, 158, "<b>Controls</b> (same model, other inputs)<br>• pure noise<br>"
      "• matched random: class means and covariances, no environmental data<br>"
      "• environment removed at test (grid mean)<br>• random pairing, shuffled labels, reversed prior",
      "link", size=XS, fill="#FFFFFF", align="left")
d.box(600, 786, 322, 74, "<b>Diagnostic only</b>: test vectors drawn with the true label (upper bound on "
      "label leakage)", "link", size=XS, fill="#FFFFFF", align="left", dashed=True)
d.arrow(pts=[(912, 580), (912, 496)], color=PURPLE, width=3)
d.save("fig3_architecture.drawio")
print("wrote fig3_architecture.drawio")
