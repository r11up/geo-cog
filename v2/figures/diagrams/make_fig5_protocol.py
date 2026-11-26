"""Figure 4 (protocol), rebuilt to mirror the original experimental-setup figure: Part A (subject-level
five-fold cross-validation, left to right from population to overall performance) and Part B
(comparison framework as a table with CONFIGURATION / DATA / COMPONENTS / FOCUS rows), with
the original titles and labels kept where still accurate. Changed:
the fold grid now shows the actual train/validation/test assignment with one colour per role;
the neural-backbone column (EEG-Conformer vs CNN, not re-run) is replaced by the environmental
content controls; the weather-station icon is replaced by the measured layers; no Singapore.
Wide layout with font sizes raised so that labels stay readable at the printed width."""
from drawio_lib import Diagram

d = Diagram("Study 1 protocol", 2000, 980)
TRAIN, VAL, TEST, FOLD = "#A9C8EA", "#CBB8E3", "#F6D36B", "#E8E8E8"
HEAD, MUTED = "#EDEDED", "#444444"
BAR, TS, B, M, S, XS = 22, 20, 19, 18, 17, 16          # bars, titles, body, medium, small, extra-small


def bar(y, label):
    d.box(8, y, 1984, 40, f"<b>{label}</b>", "grey", size=BAR, fill="#E3E3E3", rounded=False)


def line(p, q, color="#333333", w=2):
    d.arrow(pts=[p, q], end="none", color=color, width=w)


# ============================== PART A
d.box(4, 4, 1992, 520, "", "grey", fill="#FFFFFF", rounded=False)
bar(4, "PART A: SUBJECT-LEVEL FIVE-FOLD CROSS-VALIDATION")
RY, RS, RH = 160, 62, 50                                 # rows of the fold grid
MID = RY + 2 * RS + RH / 2

# --- Total study population
d.text(20, 50, 380, 60, "<b>Total Study Population</b><br>(N = 42 Subjects, Multimodal Data)", size=TS)
d.box(28, 116, 366, 392, "", "grey", fill="#FFFFFF", stroke_w=2)
d.text(28, 120, 366, 28, "<b>EEG</b>", size=TS)
for i, lab in enumerate(["30-channel<br>EEG", "DE + PSD<br>5 bands", "400 epochs<br>per subject"]):
    d.box(44 + i * 114, 152, 104, 62, lab, "neural", size=S, fill="#E4EEF8")
d.text(28, 222, 366, 28, "<b>Environmental Data</b>", size=TS)
d.box(40, 254, 108, 80, "Sentinel-2<br>(NDVI,<br>NDBI)", "env", size=S, fill="#EAF3E2")
for r in range(5):                                       # 100-cell grid, drawn as 5 x 5
    for c in range(5):
        fill = "#B9D7A0" if (r * 3 + c * 2) % 4 else "#F1D98A"
        d.box(162 + c * 14, 256 + r * 13, 14, 13, "", "env", fill=fill, stroke_w=1, rounded=False)
d.text(148, 324, 98, 24, "100-cell grid", size=XS, color=MUTED)
d.box(246, 254, 140, 80, "OpenStreetMap<br>(buildings,<br>roads)", "env", size=S, fill="#EAF3E2")
d.text(28, 352, 366, 28, "<b>Affective States</b>", size=TS)
d.box(44, 384, 332, 58, "5 EAV classes · 80 epochs<br>per class per subject", "out", size=S, fill="#FCEEE2")
d.text(28, 448, 366, 50, "30 for model selection; 12-person<br>lockbox scored once", size=S, color=MUTED)

# --- Subject-wise data split
d.text(430, 50, 250, 60, "<b>SUBJECT-WISE DATA<br>SPLIT (5 FOLDS)</b>", size=TS)
d.text(430, 116, 250, 30, "8–9 subjects per fold", size=M)
for i in range(5):
    d.box(458, RY + i * RS, 200, RH, f"Fold {i + 1}", "grey", size=TS, fill=FOLD, stroke_w=1.5)
d.text(430, 464, 250, 48, "3 random assignments<br>of subjects to folds", size=S, color=MUTED)
d.arrow(pts=[(396, MID), (454, MID)], width=2.5)
d.arrow(pts=[(660, MID), (690, MID)], width=2.5)

# --- Iterative cross-validation loop
d.text(700, 44, 520, 30, "<b>ITERATIVE CROSS-VALIDATION LOOP</b>", size=TS)
d.text(706, 76, 340, 72, "<b>TRAINING SET</b><br>(Subject-wise, four folds)", size=M, valign="top")
d.text(1032, 76, 74, 72, "<b>VAL.</b><br>(4 subj.)", size=XS, valign="top")
d.text(1102, 76, 100, 80, "<b>TEST SET</b><br>(Fold k,<br>Unseen<br>Individuals)", size=XS, valign="top")
line((694, RY + RH / 2), (694, RY + 4 * RS + RH / 2))
for k in range(5):
    y = RY + k * RS
    d.arrow(pts=[(694, y + RH / 2), (708, y + RH / 2)], width=2)
    train = [f for f in range(1, 6) if f != k + 1]
    for c, f in enumerate(train):
        d.box(710 + c * 82, y, 76, RH, f"Fold {f}", "grey", size=M, fill=TRAIN, stroke_w=1)
    d.box(1038, y, 58, RH, "val", "grey", size=XS, fill=VAL, stroke_w=1)
    d.box(1104, y, 96, RH, f"Fold {k + 1}", "grey", size=M, fill=TEST, stroke_w=1)
    line((1200, y + RH / 2), (1222, y + RH / 2))
line((1222, RY + RH / 2), (1222, RY + 4 * RS + RH / 2))
d.text(700, 464, 520, 46, "Validation: 4 subjects from the training folds,<br>for early stopping and tuning",
       size=S, color=MUTED)
d.arrow(pts=[(1222, MID), (1244, MID)], width=2.5)

# --- Model training
d.text(1244, 50, 170, 60, "<b>MODEL<br>TRAINING</b>", size=TS)
d.box(1246, 172, 168, 280, "", "grey", fill="#F0F0F0", stroke_w=1.5)
d.arrow(pts=[(1330, 112), (1330, 170)], width=2.5)
d.box(1260, 188, 140, 84, "Neural<br>Tower", "neural", size=B, fill="#E4EEF8")
d.text(1260, 274, 140, 36, "<b>+</b>", size=26)
d.box(1258, 312, 144, 84, "Environmental<br>Tower", "env", size=S, fill="#EAF3E2")
d.text(1246, 402, 168, 44, "training folds only", size=XS, color=MUTED)

# --- Performance estimation
d.text(1424, 50, 200, 60, "<b>PERFORMANCE<br>ESTIMATION</b>", size=TS)
d.arrow(pts=[(1416, MID), (1446, MID)], width=2.5)
d.box(1448, 266, 150, 86, "<b>TEST<br>SET</b>", "grey", size=BAR, fill=TEST, stroke_w=2)
d.text(1420, 356, 206, 66, "environmental input<br>drawn independently<br>of the label", size=XS, color=MUTED)

# --- Overall generalization performance
d.text(1636, 44, 350, 84, "<b>OVERALL<br>GENERALIZATION<br>PERFORMANCE</b>", size=TS)
d.arrow(pts=[(1600, MID), (1640, MID)], width=2.5)
d.box(1642, 132, 340, 262, "", "grey", fill="#FFFFFF", stroke_w=2)
d.text(1642, 138, 340, 58, "<b>Average Metrics</b><br>(e.g., F1, Accuracy)", size=TS)
line((1740, 210), (1740, 312), color="#555555", w=1.5)
line((1740, 312), (1900, 312), color="#555555", w=1.5)
for i, h in enumerate([44, 76, 62]):                     # schematic bars, no data
    d.box(1760 + i * 46, 312 - h, 30, h, "", "grey", fill="#C9C9C9", stroke_w=1, rounded=False)
d.text(1642, 318, 340, 70, "per held-out subject, mean of<br>3 repetitions, 95% bootstrap CI", size=S)

# --- legend
d.box(1430, 432, 552, 84, "", "grey", fill="#FFFFFF", stroke_w=1.5)
for i, (lab, col) in enumerate([("Training set (subject-wise)", TRAIN), ("Validation (4 subjects)", VAL),
                                 ("Test set (subject-wise, fold k)", TEST), ("Fold (before assignment)", FOLD)]):
    x, y = 1442 + (i % 2) * 272, 444 + (i // 2) * 36
    d.box(x, y, 30, 24, "", "grey", fill=col, stroke_w=1, rounded=False)
    d.text(x + 36, y - 4, 236, 32, lab, size=S, align="left")

# ============================== PART B
YB = 532
d.box(4, YB, 1992, 444, "", "grey", fill="#FFFFFF", rounded=False)
bar(YB, "PART B: COMPARISON FRAMEWORK AND ABLATION CONDITIONS")
Y0, Y1, Y2, Y3, Y4 = 580, 624, 702, 890, 968           # group title, configuration, components, focus, end
RHX, RHW = 12, 196
for (ya, yb, lab) in [(Y1, Y2, "CONFIGURATION"), (Y2, Y3, "DATA /<br>COMPONENTS"), (Y3, Y4, "FOCUS")]:
    d.box(RHX, ya, RHW, yb - ya, f"<b>{lab}</b>", "grey", size=M, fill=HEAD, stroke_w=1, rounded=False)

groups = [  # (title, sub-column width, [configuration labels], focus)
    ("ISOLATING ALIGNMENT STRATEGY", 240,
     ["<b>Baseline</b><br>(EEG only)", "<b>Full Multimodal</b><br>(label-free pairing, proposed)"],
     ["Focus: Reference for<br>unimodal performance",
      "Focus: Contribution of an environmental input drawn without the label"]),
    ("ISOLATING ATTENTION DIRECTIONALITY", 228,
     ["<b>Unidirectional Attention</b><br>(EEG queries environment)", "<b>Bidirectional Attention</b><br>(each queries the other)"],
     ["Focus: Direction of cross-modal<br>information flow"]),
    ("ISOLATING THE PRIOR (PECM, ORIGINAL DESIGN)", 203,
     ["<b>Pure-Noise<br>Input</b>", "<b>Matched<br>Random Input</b>", "<b>Removed<br>at Test</b>", "<b>Prior<br>Ablations</b>"],
     ["Focus: Whether the prior's effect needs environmental content,<br>or only an input tied to the label"]),
]
x = RHX + RHW + 8
cells = []
for title, w, configs, focus in groups:
    gw = w * len(configs)
    d.box(x, Y0, gw, Y1 - Y0, f"<b>{title}</b>", "grey", size=B, fill=HEAD, stroke_w=1, rounded=False)
    for j, cfg in enumerate(configs):
        cx = x + j * w
        d.box(cx, Y1, w, Y2 - Y1, cfg, "grey", size=M, fill="#FFFFFF", stroke_w=1, rounded=False)
        d.box(cx, Y2, w, Y3 - Y2, "", "grey", fill="#FFFFFF", stroke_w=1, rounded=False)
        cells.append(cx)
    fw = gw / len(focus)
    for j, f in enumerate(focus):
        d.box(x + j * fw, Y3, fw, Y4 - Y3, f, "grey", size=M, fill="#FFFFFF", stroke_w=1, rounded=False)
    x += gw + 8


def tower(cx, w, y, kind, lab, size=S, h=58, fill=None):
    return d.box(cx + (w - 170) / 2, y, 170, h, lab, kind, size=size,
                 fill=fill or {"neural": "#E4EEF8", "env": "#EAF3E2", "out": "#FCE8B8"}[kind])


# alignment strategy
w = 240
c = cells[0]
n = tower(c, w, Y2 + 16, "neural", "Neural Tower<br>(EEG tower)", h=60)
s = tower(c, w, Y2 + 112, "out", "Softmax Head", h=52)
d.arrow(n, s, exit=(0.5, 1), entry=(0.5, 0), width=2)
c = cells[1]
n = d.box(c + 12, Y2 + 12, 104, 58, "Neural<br>Tower", "neural", size=S, fill="#E4EEF8")
e = d.box(c + 124, Y2 + 12, 104, 58, "Env.<br>Tower", "env", size=S, fill="#EAF3E2")
a = d.box(c + 20, Y2 + 86, 200, 44, "Bidirectional Attention", "out", size=XS, fill="#F8D3C3")
s = d.box(c + 50, Y2 + 142, 140, 38, "Softmax Head", "out", size=XS, fill="#FCE8B8")
d.arrow(n, a, exit=(0.5, 1), entry=(44 / 200, 0), width=1.8)
d.arrow(e, a, exit=(0.5, 1), entry=(156 / 200, 0), width=1.8)
d.arrow(a, s, exit=(0.5, 1), entry=(0.5, 0), width=1.8)

# attention directionality
w = 228
for k, both in [(2, False), (3, True)]:
    c = cells[k]
    d.box(c + 22, Y2 + 16, 74, 156, "Neural<br>Tower", "neural", size=S, fill="#E4EEF8")
    d.box(c + w - 96, Y2 + 16, 74, 156, "Env.<br>Tower", "env", size=S, fill="#EAF3E2")
    d.arrow(pts=[(c + 98, Y2 + 70), (c + w - 98, Y2 + 70)], width=2)
    if both:
        d.arrow(pts=[(c + w - 98, Y2 + 118), (c + 98, Y2 + 118)], width=2)

# environmental content
w = 203
inputs = [("Random vectors,<br>no environmental<br>data", True),
          ("Synthetic vectors<br>with the prior's class<br>means, covariances", True),
          ("Training: prior<br>Test: grid mean", False),
          ("Reversed prior ·<br>random pairing ·<br>shuffled labels", False)]
for k, (lab, dashed) in enumerate(inputs):
    c = cells[4 + k]
    i = d.box(c + 12, Y2 + 14, w - 24, 80, lab, "link", size=XS, fill="#EFE8F6", dashed=dashed)
    e = d.box(c + 22, Y2 + 116, w - 44, 56, "Environmental<br>Tower", "env", size=XS, fill="#EAF3E2")
    d.arrow(i, e, exit=(0.5, 1), entry=(0.5, 0), width=1.8)

d.save("fig5_protocol.drawio")
print("wrote fig5_protocol.drawio")
