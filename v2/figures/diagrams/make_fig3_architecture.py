"""Figure 3: the Study 1 dual-tower architecture, drawn with real-data graphics (EAV EEG traces,
feature map and covariance of one epoch; the Sentinel-2/OSM layers and graph of the Astana grid)
and icons. How each epoch's environmental vector is drawn sits in the environmental branch, just
before the graph tower, and the controls sit under the output. Compact canvas so that text and
graphics stay legible at the width of the printed page."""
from pathlib import Path

from drawio_lib import Diagram

A = Path(__file__).resolve().parents[1] / "graphics" / "assets"
d = Diagram("Dual-tower architecture (graphical)", 940, 692)
BLUE, GREEN, PURPLE, ORANGE, INK, MUTED = "#4A7FB5", "#6A9F4B", "#8064A2", "#D9822B", "#1F2328", "#5B6270"
T, L, S, XS = 20, 18, 16, 15                                   # title, label, small, extra-small font sizes
CLS = [("neutral", "neutral"), ("sadness", "sad"), ("anger", "angry"), ("happiness", "happy"), ("calmness", "calm")]
TOWER_Y, EMB_Y = 498, 634                                      # both towers and both embeddings share a row


def label(x, y, w, text, size=S, color=INK, h=26, align="center"):
    return d.text(x, y, w, h, text, size=size, color=color, align=align)


def vec(cx, y, n, color, w=12, h=18):
    """An embedding drawn as a row of cells centred on cx."""
    x0 = cx - n * w / 2
    for i in range(n):
        d.box(x0 + i * w, y, w, h, "", "grey", fill=color, stroke_w=0.8, rounded=False)


# ======================= EEG branch
d.panel(8, 8, 316, 676, "EEG branch", "neural", size=T)
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
label(130, 294, 160, "band covariances<br>(5 bands)", S, h=40)
d.arrow(pts=[(166, 340), (166, 368)], color=BLUE, width=3)
d.box(20, 372, 292, 100, "<b>Per participant, without labels</b>: covariances re-centred at the participant's "
      "Riemannian mean, then mapped to the tangent space", "neural", size=XS, fill="#F3F7FC")
d.arrow(pts=[(166, 474), (166, TOWER_Y - 4)], color=BLUE, width=3)
d.box(20, TOWER_Y, 292, 118, "", "neural", fill="#F3F7FC")
label(20, TOWER_Y + 4, 292, "<b>EEG tower</b>", L)
for i in range(30):
    d.box(52 + i * 7, TOWER_Y + 34, 6, 22, "", "grey", fill="#A9C8EA", stroke_w=0.5, rounded=False)
d.box(52 + 30 * 7 + 8, TOWER_Y + 34, 16, 22, "", "grey", fill="#F6C28B", stroke_w=0.5, rounded=False)
label(20, TOWER_Y + 58, 292, "30 channel tokens + covariance token", XS, MUTED, h=22)
d.box(74, TOWER_Y + 84, 184, 30, "2-layer transformer", "neural", size=S, fill="#E4EEF8")
d.arrow(pts=[(166, TOWER_Y + 120), (166, EMB_Y - 2)], color=BLUE, width=3)
vec(166, EMB_Y, 12, BLUE)
label(20, EMB_Y + 18, 292, "EEG embedding <i>z</i><sub>EEG</sub>", 17, h=24)

# ======================= Environmental branch
d.panel(616, 8, 316, 676, "Environmental branch (Astana)", "env", size=T)
d.image(622, 58, 52, 52, A / "satellite.svg")
d.image(676, 48, 104, 104, A / "astana_rgb_grid.png")
label(668, 152, 120, "Sentinel-2", S, h=22)
d.image(804, 48, 104, 104, A / "astana_osm.png")
label(790, 152, 132, "OpenStreetMap", S, h=22)
d.arrow(pts=[(774, 176), (774, 192)], color=GREEN, width=3)
d.image(630, 196, 80, 80, A / "grid_ndvi.png")
d.image(716, 196, 80, 80, A / "grid_building.png")
label(622, 278, 180, "100 cells × 4 layers", S, h=22)
d.image(824, 194, 84, 84, A / "grid_graph.png")
label(806, 278, 120, "cell graph", S, h=22)
d.arrow(pts=[(774, 301), (774, 316)], color=GREEN, width=3)
d.box(628, 318, 292, 158, "", "link", fill="#F7F3FB")
label(628, 321, 292, "<b>Each epoch's <i>v</i><sub>env</sub></b>", 17, h=24)
d.text(636, 346, 280, 128,
       "<b>PECM</b> (under test), training only, by label:<br>"
       f"<font color='{ORANGE}'>●</font> anger, sadness: exposure, top quartile<br>"
       f"<font color='{GREEN}'>●</font> happiness, calm: NDVI, top quartile<br>"
       "<font color='#8C8C8C'>●</font> neutral: interquartile of both<br>"
       "<b>Label-free</b>: all cells, without the label<br>"
       "<b>At test</b>: always without the label;<br>with it only in a diagnostic run", size=XS, align="left")
d.arrow(pts=[(774, 478), (774, TOWER_Y - 4)], color=GREEN, width=3)
d.box(628, TOWER_Y, 292, 118, "", "env", fill="#F4F9F0")
label(628, TOWER_Y + 4, 292, "<b>Graph tower</b>", L)
for i in range(8):
    d.box(638 + i * 36, TOWER_Y + 34, 24, 24, "", "env", fill="#CFE3C2", stroke_w=1)
    if i < 7:
        d.arrow(pts=[(662 + i * 36, TOWER_Y + 46), (674 + i * 36, TOWER_Y + 46)], color="#8FB57A", width=1.5,
                end="none")
label(628, TOWER_Y + 60, 292, "<i>v</i><sub>env</sub> added to every node", XS, MUTED, h=22)
d.box(682, TOWER_Y + 84, 184, 30, "2 GCN layers + GRU", "env", size=S, fill="#E6F1DE")
d.arrow(pts=[(774, TOWER_Y + 120), (774, EMB_Y - 2)], color=GREEN, width=3)
vec(774, EMB_Y, 12, GREEN)
label(640, EMB_Y + 18, 268, "environmental embedding <i>z</i><sub>ENV</sub>", 17, h=24)

# ======================= Fusion (centre)
d.box(342, 22, 250, 72, "alignment loss<br>‖<i>z</i><sub>EEG</sub> − <i>z</i><sub>ENV</sub>‖², λ = 0.1", "link",
      size=S, fill="#F4EFFA", dashed=True)
d.arrow(pts=[(467, 96), (467, 120)], color=PURPLE, width=2, dashed=True)
d.box(332, 124, 270, 212, "", "grey", fill="#F6F6F6", stroke_w=1.5)
label(332, 128, 270, "<b>Bidirectional<br>cross-modal attention</b>", L, h=50)
d.box(348, 182, 46, 140, "<i>z</i><sub>EEG</sub>", "neural", size=S, fill="#A9C8EA")
d.box(540, 182, 46, 140, "<i>z</i><sub>ENV</sub>", "env", size=S, fill="#BFD9AE")
label(396, 190, 142, "EEG → environment", XS)
d.arrow(pts=[(396, 218), (538, 218)], color=BLUE, width=3)
d.arrow(pts=[(538, 286), (396, 286)], color=GREEN, width=3)
label(396, 292, 142, "environment → EEG", XS)
d.arrow(pts=[(240, EMB_Y + 9), (348, 252)], via=[(328, EMB_Y + 9), (328, 252)], color=BLUE, width=3)
d.arrow(pts=[(700, EMB_Y + 9), (586, 252)], via=[(609, EMB_Y + 9), (609, 252)], color=GREEN, width=3)
d.arrow(pts=[(467, 338), (467, 354)], color=ORANGE, width=3)
d.box(342, 356, 250, 44, "concatenate → MLP → softmax", "out", size=S, fill="#FCEEE2")
d.arrow(pts=[(467, 402), (467, 416)], color=ORANGE, width=3)
for i, (name, short) in enumerate(CLS):
    x = 336 + i * 54
    d.image(x + 2, 418, 46, 46, A / f"face_{name}.svg")
    label(x - 2, 466, 54, short, XS, h=22)
label(332, 488, 270, "<b>five affective states</b>", S, h=24)
d.box(342, 526, 250, 158, "<b>Controls</b> (same model, other <i>v</i><sub>env</sub>)<br>• pure noise<br>"
      "• matched random: class means and covariances, no environmental data<br>"
      "• removed at test (grid mean)<br>• label-free pairing; shuffled labels; reversed prior",
      "link", size=XS, fill="#FFFFFF", align="left")
d.save("fig3_architecture.drawio")
print("wrote fig3_architecture.drawio")
