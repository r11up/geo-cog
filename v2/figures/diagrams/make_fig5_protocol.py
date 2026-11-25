"""Figure 5, rebuilt to mirror the original experimental-setup figure: Part A (subject-level
five-fold cross-validation, left to right from population to overall performance) and Part B
(comparison framework as a table with CONFIGURATION / DATA / COMPONENTS / FOCUS rows), with
the original titles and labels kept where still accurate. Changed:
the fold grid now shows the actual train/validation/test assignment with one colour per role;
the neural-backbone column (EEG-Conformer vs CNN, not re-run) is replaced by the environmental
content controls; the weather-station icon is replaced by the measured layers; no Singapore."""
from drawio_lib import Diagram

d = Diagram("Study 1 protocol", 2000, 940)
TRAIN, VAL, TEST, FOLD = "#A9C8EA", "#CBB8E3", "#F6D36B", "#E8E8E8"
HEAD = "#EDEDED"


def bar(y, label):
    d.box(8, y, 1984, 36, f"<b>{label}</b>", "grey", size=19, fill="#E3E3E3", rounded=False)


def line(p, q, color="#333333", w=2):
    d.arrow(pts=[p, q], end="none", color=color, width=w)


# ============================== PART A
d.box(4, 4, 1992, 466, "", "grey", fill="#FFFFFF", rounded=False)
bar(4, "PART A: SUBJECT-LEVEL FIVE-FOLD CROSS-VALIDATION")

# --- Total study population
d.text(20, 46, 380, 56, "<b>Total Study Population</b><br>(N = 42 Subjects, with Multi-modal Data)", size=17)
pop = d.box(28, 108, 366, 350, "", "grey", fill="#FFFFFF", stroke_w=2)
d.text(28, 112, 366, 26, "<b>EEG</b>", size=17)
for i, lab in enumerate(["30-channel<br>EEG", "DE + PSD<br>5 bands", "400 epochs<br>per subject"]):
    d.box(44 + i * 114, 142, 104, 58, lab, "neural", size=14, fill="#E4EEF8")
d.text(28, 208, 366, 26, "<b>Environmental Data</b>", size=17)
d.box(44, 240, 104, 66, "Sentinel-2<br>(NDVI,<br>NDBI)", "env", size=14, fill="#EAF3E2")
for r in range(5):                                       # 100-cell grid, drawn as 5 x 5
    for c in range(5):
        fill = "#B9D7A0" if (r * 3 + c * 2) % 4 else "#F1D98A"
        d.box(166 + c * 14, 242 + r * 13, 14, 13, "", "env", fill=fill, stroke_w=1, rounded=False)
d.text(150, 310, 110, 22, "100-cell grid", size=13, color="#444444")
d.box(272, 240, 104, 66, "OpenStreet-<br>Map (build-<br>ings, roads)", "env", size=14, fill="#EAF3E2")
d.text(28, 334, 366, 24, "<b>Affective States</b>", size=17)
d.box(44, 362, 332, 44, "5 EAV classes · 80 epochs per class per subject", "out", size=14, fill="#FCEEE2")
d.text(28, 414, 366, 34, "Data vectors", size=17)

# --- Subject-wise data split
d.text(430, 46, 250, 56, "<b>SUBJECT-WISE DATA<br>SPLIT (5 FOLDS)</b>", size=17)
d.text(430, 104, 250, 26, "8–9 subjects per fold", size=15)
for i in range(5):
    d.box(458, 140 + i * 56, 200, 44, f"Fold {i + 1}", "grey", size=17, fill=FOLD, stroke_w=1.5)
d.text(430, 422, 250, 40, "3 random assignments<br>of subjects to folds", size=14, color="#444444")
d.arrow(pts=[(396, 283), (454, 283)], width=2.5)
d.arrow(pts=[(660, 283), (690, 283)], width=2.5)

# --- Iterative cross-validation loop
d.text(700, 46, 520, 30, "<b>ITERATIVE CROSS-VALIDATION LOOP</b>", size=17)
d.text(706, 80, 340, 52, "<b>TRAINING SET</b><br>(Subject-wise, four folds)", size=15)
d.text(1046, 74, 64, 58, "<b>VAL.</b><br>(4 subj.)", size=13)
d.text(1110, 74, 120, 64, "<b>TEST SET</b><br>(Fold k, Unseen<br>Individuals)", size=14)
RY, RH = 140, 44
line((694, RY + RH / 2), (694, RY + 4 * 56 + RH / 2))
for k in range(5):
    y = RY + k * 56
    d.arrow(pts=[(694, y + RH / 2), (708, y + RH / 2)], width=2)
    train = [f for f in range(1, 6) if f != k + 1]
    for c, f in enumerate(train):
        d.box(710 + c * 84, y, 78, RH, f"Fold {f}", "grey", size=15, fill=TRAIN, stroke_w=1)
    d.box(1046, y, 58, RH, "val", "grey", size=13, fill=VAL, stroke_w=1)
    d.box(1112, y, 96, RH, f"Fold {k + 1}", "grey", size=15, fill=TEST, stroke_w=1)
    line((1208, y + RH / 2), (1222, y + RH / 2))
line((1222, RY + RH / 2), (1222, RY + 4 * 56 + RH / 2))
d.text(700, 422, 520, 40, "Validation: 4 subjects from the training folds, for early stopping only", size=14, color="#444444")
d.arrow(pts=[(1222, 283), (1244, 283)], width=2.5)

# --- Model training
d.text(1244, 46, 170, 56, "<b>MODEL<br>TRAINING</b>", size=17)
d.box(1246, 160, 168, 250, "", "grey", fill="#F0F0F0", stroke_w=1.5)
d.arrow(pts=[(1330, 104), (1330, 158)], width=2.5)
d.box(1262, 180, 136, 72, "Neural<br>Tower", "neural", size=16, fill="#E4EEF8")
d.text(1262, 256, 136, 30, "<b>+</b>", size=22)
d.box(1262, 292, 136, 72, "Environmental<br>Tower", "env", size=15, fill="#EAF3E2")
d.text(1246, 370, 168, 36, "training folds only", size=14, color="#444444")

# --- Performance estimation
d.text(1430, 46, 190, 56, "<b>PERFORMANCE<br>ESTIMATION</b>", size=17)
d.arrow(pts=[(1416, 283), (1446, 283)], width=2.5)
d.box(1448, 246, 150, 74, "<b>TEST<br>SET</b>", "grey", size=19, fill=TEST, stroke_w=2)
d.text(1428, 322, 190, 44, "environmental input drawn<br>independently of the label", size=13, color="#444444")

# --- Overall generalization performance
d.text(1636, 40, 350, 80, "<b>OVERALL<br>GENERALIZATION<br>PERFORMANCE</b>", size=17)
d.arrow(pts=[(1600, 283), (1640, 283)], width=2.5)
d.box(1642, 124, 340, 236, "", "grey", fill="#FFFFFF", stroke_w=2)
d.text(1642, 130, 340, 54, "<b>Average Metrics</b><br>(e.g., F1, Accuracy)", size=17)
line((1740, 196), (1740, 300), color="#555555", w=1.5)
line((1740, 300), (1900, 300), color="#555555", w=1.5)
for i, h in enumerate([44, 76, 62]):                     # schematic bars, no data
    d.box(1760 + i * 46, 300 - h, 30, h, "", "grey", fill="#C9C9C9", stroke_w=1, rounded=False)
d.text(1642, 306, 340, 50, "per held-out subject, mean of<br>3 repetitions, 95% bootstrap CI", size=14)

# --- legend
d.box(1430, 380, 552, 80, "", "grey", fill="#FFFFFF", stroke_w=1.5)
for i, (lab, col) in enumerate([("Training set (subject-wise)", TRAIN), ("Validation (4 subjects)", VAL),
                                 ("Test set (subject-wise, fold k)", TEST), ("Fold (before assignment)", FOLD)]):
    x, y = 1444 + (i % 2) * 276, 390 + (i // 2) * 34
    d.box(x, y, 30, 24, "", "grey", fill=col, stroke_w=1, rounded=False)
    d.text(x + 36, y - 3, 236, 30, lab, size=14, align="left")

# ============================== PART B
d.box(4, 478, 1992, 456, "", "grey", fill="#FFFFFF", rounded=False)
bar(478, "PART B: COMPARISON FRAMEWORK AND ABLATION CONDITIONS")
Y0, Y1, Y2, Y3, Y4 = 520, 570, 642, 822, 926          # group title, configuration, components, focus, end
RHX, RHW = 12, 196
for (ya, yb, lab) in [(Y1, Y2, "CONFIGURATION"), (Y2, Y3, "DATA /<br>COMPONENTS"), (Y3, Y4, "FOCUS")]:
    d.box(RHX, ya, RHW, yb - ya, f"<b>{lab}</b>", "grey", size=16, fill=HEAD, stroke_w=1, rounded=False)

groups = [  # (title, sub-column width, [configuration labels], focus)
    ("ISOLATING ALIGNMENT STRATEGY", 250,
     ["<b>Baseline</b><br>(EEG only)", "<b>Full Multimodal</b><br>(bidirectional)"],
     ["Focus: Reference for<br>unimodal performance", "Focus: Contribution of<br>probabilistically aligned context"]),
    ("ISOLATING ATTENTION DIRECTIONALITY", 216,
     ["<b>Unidirectional Attention</b><br>(EEG queries environment)", "<b>Bidirectional Attention</b><br>(each queries the other)"],
     ["Focus: Direction of cross-modal<br>information flow"]),
    ("ISOLATING ENVIRONMENTAL CONTENT", 203,
     ["<b>Pure-Noise<br>Input</b>", "<b>Matched<br>Random Input</b>", "<b>Removed<br>at Test</b>", "<b>Prior<br>Ablations</b>"],
     ["Focus: Whether any gain needs environmental content,<br>or only an extra structured input"]),
]
x = RHX + RHW + 8
cells = []
for title, w, configs, focus in groups:
    gw = w * len(configs)
    d.box(x, Y0, gw, Y1 - Y0, f"<b>{title}</b>", "grey", size=17, fill=HEAD, stroke_w=1, rounded=False)
    for j, cfg in enumerate(configs):
        cx = x + j * w
        d.box(cx, Y1, w, Y2 - Y1, cfg, "grey", size=16, fill="#FFFFFF", stroke_w=1, rounded=False)
        d.box(cx, Y2, w, Y3 - Y2, "", "grey", fill="#FFFFFF", stroke_w=1, rounded=False)
        cells.append(cx)
    fw = gw / len(focus)
    for j, f in enumerate(focus):
        d.box(x + j * fw, Y3, fw, Y4 - Y3, f, "grey", size=15, fill="#FFFFFF", stroke_w=1, rounded=False)
    x += gw + 8


def tower(cx, w, y, kind, lab, size=14, h=48, fill=None):
    return d.box(cx + (w - 150) / 2, y, 150, h, lab, kind, size=size,
                 fill=fill or {"neural": "#E4EEF8", "env": "#EAF3E2", "out": "#FCE8B8"}[kind])


# alignment strategy
w = 250
c = cells[0]
n = tower(c, w, 664, "neural", "Neural Tower<br>(EEG tower)")
s = tower(c, w, 758, "out", "Softmax Head")
d.arrow(n, s, exit=(0.5, 1), entry=(0.5, 0), width=2)
c = cells[1]
n = d.box(c + 12, 652, 108, 50, "Neural<br>Tower", "neural", size=14, fill="#E4EEF8")
e = d.box(c + 130, 652, 108, 50, "Environmental<br>Tower", "env", size=13, fill="#EAF3E2")
a = d.box(c + 40, 718, 170, 40, "Bidirectional Attention", "out", size=13, fill="#F8D3C3")
s = d.box(c + 65, 774, 120, 38, "Softmax Head", "out", size=13, fill="#FCE8B8")
d.arrow(n, a, exit=(0.5, 1), entry=(26 / 170, 0), width=1.8)
d.arrow(e, a, exit=(0.5, 1), entry=(144 / 170, 0), width=1.8)
d.arrow(a, s, exit=(0.5, 1), entry=(0.5, 0), width=1.8)

# attention directionality
w = 216
for k, both in [(2, False), (3, True)]:
    c = cells[k]
    n = d.box(c + 22, 660, 72, 140, "Neural<br>Tower", "neural", size=14, fill="#E4EEF8")
    e = d.box(c + w - 94, 660, 72, 140, "Env.<br>Tower", "env", size=14, fill="#EAF3E2")
    d.arrow(pts=[(c + 96, 708), (c + w - 96, 708)], width=2)
    if both:
        d.arrow(pts=[(c + w - 96, 754), (c + 96, 754)], width=2)

# environmental content
w = 203
inputs = [("Random vectors,<br>no environmental<br>data", True),
          ("Synthetic vectors<br>with the prior's class<br>means, covariances", True),
          ("Training: prior<br>Test: grid mean", False),
          ("Reversed prior ·<br>random pairing ·<br>shuffled labels", False)]
for k, (lab, dashed) in enumerate(inputs):
    c = cells[4 + k]
    i = d.box(c + 14, 654, w - 28, 72, lab, "link", size=13, fill="#EFE8F6", dashed=dashed)
    e = d.box(c + 26, 760, w - 52, 48, "Environmental<br>Tower", "env", size=13, fill="#EAF3E2")
    d.arrow(i, e, exit=(0.5, 1), entry=(0.5, 0), width=1.8)

d.save("fig5_protocol.drawio")
print("wrote fig5_protocol.drawio")
