"""Figure 4: the Study 1 protocol, mirroring the original experimental-setup figure. Part A (subject-level
five-fold cross-validation, from the study population to overall performance) and Part B (comparison
framework as a table with CONFIGURATION / DATA / COMPONENTS / FOCUS rows). Each part is laid out in two
rows on a narrow canvas so that the text stays legible at the printed width; the second row of Part A
reads from right to left, following the arrows. Changed from the original: the fold grid shows the
actual train/validation/test assignment with one colour per role; the neural-backbone column is
replaced by the environmental-content controls; the measured layers replace the weather-station icon."""
from drawio_lib import Diagram

W, H = 1000, 1316
d = Diagram("Study 1 protocol", W, H)
TRAIN, VAL, TEST, FOLD, HEAD = "#A9C8EA", "#CBB8E3", "#F6D36B", "#E8E8E8", "#EDEDED"
MUTED = "#444444"
BAR, TS, B, S, RF = 19, 17, 16, 15, 14          # part bars, stage titles, body, small, row headers


def bar(y, label):
    d.box(8, y, W - 16, 34, f"<b>{label}</b>", "grey", size=BAR, fill="#E3E3E3", rounded=False)


def line(p, q, color="#333333", w=2):
    d.arrow(pts=[p, q], end="none", color=color, width=w)


def title(x, y, w, h, html):
    d.text(x, y, w, h, f"<b>{html}</b>", size=TS)


# ============================== PART A
d.box(4, 4, W - 8, 666, "", "grey", fill="#FFFFFF", rounded=False)
bar(4, "PART A: SUBJECT-LEVEL FIVE-FOLD CROSS-VALIDATION")

# --- Total study population
d.text(10, 42, 300, 50, "<b>Total Study Population</b><br>(N = 42 subjects, multimodal data)", size=TS)
d.box(14, 96, 292, 336, "", "grey", fill="#FFFFFF", stroke_w=2)
d.text(14, 100, 292, 24, "<b>EEG</b>", size=B)
for i, lab in enumerate(["30-channel<br>EEG", "DE + PSD<br>5 bands", "band<br>covariances"]):
    d.box(18 + i * 95, 126, 92, 54, lab, "neural", size=S, fill="#E4EEF8")
d.text(14, 186, 292, 24, "<b>Environmental Data</b>", size=B)
d.box(18, 214, 82, 66, "Sentinel-2<br>(NDVI,<br>NDBI)", "env", size=S, fill="#EAF3E2")
for r in range(5):                                       # 100-cell grid, drawn as 5 x 5
    for c in range(5):
        fill = "#B9D7A0" if (r * 3 + c * 2) % 4 else "#F1D98A"
        d.box(108 + c * 12, 216 + r * 12, 12, 12, "", "env", fill=fill, stroke_w=1, rounded=False)
d.text(92, 282, 92, 22, "100-cell grid", size=S, color=MUTED)
d.box(176, 214, 124, 66, "OpenStreetMap<br>(buildings,<br>roads)", "env", size=S, fill="#EAF3E2")
d.text(14, 306, 292, 24, "<b>Affective States</b>", size=B)
d.box(22, 334, 276, 48, "5 EAV classes · 80 epochs<br>per class per subject", "out", size=S, fill="#FCEEE2")
d.text(14, 386, 292, 42, "30 for model selection; 12-person<br>lockbox scored once", size=S, color=MUTED)

# --- Subject-wise data split
title(316, 40, 152, 66, "SUBJECT-WISE<br>DATA SPLIT<br>(5 FOLDS)")
d.text(316, 106, 152, 40, "8–9 subjects<br>per fold", size=S)
for i in range(5):
    d.box(334, 150 + i * 50, 116, 40, f"Fold {i + 1}", "grey", size=B, fill=FOLD, stroke_w=1.5)
d.text(316, 396, 152, 56, "3 random<br>assignments of<br>subjects to folds", size=S, color=MUTED)
d.arrow(pts=[(308, 270), (332, 270)], width=2.5)

# --- Iterative cross-validation loop
title(476, 38, 480, 28, "ITERATIVE CROSS-VALIDATION LOOP")
d.text(496, 66, 288, 80, "<b>TRAINING SET</b><br>(subject-wise, four folds)", size=S, valign="top")
d.text(780, 66, 70, 80, "<b>VAL.</b><br>(4 subj.)", size=S, valign="top")
d.text(831, 66, 120, 80, "<b>TEST SET</b><br>(fold k,<br>unseen<br>individuals)", size=S, valign="top")
RY, RS, RH = 150, 50, 40
line((482, RY + RH / 2), (482, RY + 4 * RS + RH / 2))
d.arrow(pts=[(452, 270), (480, 270)], width=2.5)
for k in range(5):
    y = RY + k * RS
    d.arrow(pts=[(482, y + RH / 2), (494, y + RH / 2)], width=2)
    train = [f for f in range(1, 6) if f != k + 1]
    for c, f in enumerate(train):
        d.box(496 + c * 72, y, 68, RH, f"Fold {f}", "grey", size=B, fill=TRAIN, stroke_w=1)
    d.box(788, y, 54, RH, "val", "grey", size=S, fill=VAL, stroke_w=1)
    d.box(846, y, 90, RH, f"Fold {k + 1}", "grey", size=B, fill=TEST, stroke_w=1)
    line((936, y + RH / 2), (950, y + RH / 2))
line((950, RY + RH / 2), (950, RY + 4 * RS + RH / 2))
d.text(476, 396, 460, 44, "Validation: 4 subjects from the training folds,<br>for early stopping and tuning",
       size=S, color=MUTED)
d.arrow(pts=[(950, RY + 4 * RS + RH / 2), (950, 512)], width=2.5)

# --- Model training (second row, read from right to left)
title(676, 466, 314, 44, "MODEL TRAINING")
d.box(676, 514, 314, 146, "", "grey", fill="#F0F0F0", stroke_w=1.5)
d.box(690, 524, 132, 60, "Neural<br>Tower", "neural", size=B, fill="#E4EEF8")
d.text(822, 534, 22, 40, "<b>+</b>", size=22)
d.box(844, 524, 132, 60, "Environmental<br>Tower", "env", size=B, fill="#EAF3E2")
d.text(680, 590, 306, 64, "training folds only; environmental input drawn with the label (PECM) "
       "or without it (proposed)", size=S, color=MUTED)

# --- Performance estimation
title(410, 466, 246, 44, "PERFORMANCE<br>ESTIMATION")
d.box(463, 540, 140, 72, "<b>TEST<br>SET</b>", "grey", size=20, fill=TEST, stroke_w=2)
d.text(410, 616, 246, 44, "environmental input drawn<br>independently of the label", size=S, color=MUTED)
d.arrow(pts=[(674, 576), (605, 576)], width=2.5)

# --- Overall generalization performance
title(14, 466, 376, 44, "OVERALL GENERALIZATION<br>PERFORMANCE")
d.box(14, 514, 376, 146, "", "grey", fill="#FFFFFF", stroke_w=2)
d.text(14, 520, 376, 28, "<b>Average Metrics</b> (e.g., F1, Accuracy)", size=B)
line((52, 560), (52, 644), color="#555555", w=1.5)
line((52, 644), (172, 644), color="#555555", w=1.5)
for i, h in enumerate([40, 68, 54]):                     # schematic bars, no data
    d.box(66 + i * 36, 644 - h, 26, h, "", "grey", fill="#C9C9C9", stroke_w=1, rounded=False)
d.text(184, 560, 200, 84, "per held-out subject,<br>mean of 3 repetitions,<br>95% bootstrap CI", size=S)
d.arrow(pts=[(461, 576), (392, 576)], width=2.5)

# ============================== PART B
YB = 680
d.box(4, YB, W - 8, 630, "", "grey", fill="#FFFFFF", rounded=False)
bar(YB, "PART B: COMPARISON FRAMEWORK AND ABLATION CONDITIONS")
RHX, RHW = 12, 138


def row_heads(y1, y2, y3, y4):
    for ya, yb, lab in [(y1, y2, "CONFIGURATION"), (y2, y3, "DATA /<br>COMPONENTS"), (y3, y4, "FOCUS")]:
        d.box(RHX, ya, RHW, yb - ya, f"<b>{lab}</b>", "grey", size=RF, fill=HEAD, stroke_w=1, rounded=False)


def group(x, ys, name, w, configs, focus):
    y0, y1, y2, y3, y4 = ys
    gw = w * len(configs)
    d.box(x, y0, gw, y1 - y0, f"<b>{name}</b>", "grey", size=B, fill=HEAD, stroke_w=1, rounded=False)
    cells = []
    for j, cfg in enumerate(configs):
        cx = x + j * w
        d.box(cx, y1, w, y2 - y1, cfg, "grey", size=B, fill="#FFFFFF", stroke_w=1, rounded=False)
        d.box(cx, y2, w, y3 - y2, "", "grey", fill="#FFFFFF", stroke_w=1, rounded=False)
        cells.append(cx)
    fw = gw / len(focus)
    for j, f in enumerate(focus):
        d.box(x + j * fw, y3, fw, y4 - y3, f, "grey", size=B, fill="#FFFFFF", stroke_w=1, rounded=False)
    return cells


# --- first row: alignment strategy and attention directionality
ys = (722, 756, 826, 962, 1026)
row_heads(*ys[1:])
w = 206
ca = group(156, ys, "ISOLATING ALIGNMENT STRATEGY", w,
           ["<b>Baseline</b><br>(EEG only)", "<b>Full Multimodal</b><br>(label-free pairing, proposed)"],
           ["Focus: Reference for<br>unimodal performance",
            "Focus: Contribution of an environmental input drawn without the label"])
cd = group(156 + 2 * w + 6, ys, "ISOLATING ATTENTION DIRECTIONALITY", w,
           ["<b>Unidirectional Attention</b><br>(EEG queries environment)",
            "<b>Bidirectional Attention</b><br>(each queries the other)"],
           ["Focus: Direction of cross-modal information flow"])
Y2 = ys[2]
c = ca[0]
n = d.box(c + 28, Y2 + 14, 150, 46, "Neural Tower<br>(EEG tower)", "neural", size=S, fill="#E4EEF8")
s = d.box(c + 28, Y2 + 84, 150, 40, "Softmax Head", "out", size=S, fill="#FCE8B8")
d.arrow(n, s, exit=(0.5, 1), entry=(0.5, 0), width=2)
c = ca[1]
n = d.box(c + 10, Y2 + 8, 90, 44, "Neural<br>Tower", "neural", size=S, fill="#E4EEF8")
e = d.box(c + 106, Y2 + 8, 90, 44, "Env.<br>Tower", "env", size=S, fill="#EAF3E2")
a = d.box(c + 13, Y2 + 64, 180, 30, "Bidirectional Attention", "out", size=S, fill="#F8D3C3")
s = d.box(c + 43, Y2 + 104, 120, 28, "Softmax Head", "out", size=S, fill="#FCE8B8")
d.arrow(n, a, exit=(0.5, 1), entry=(42 / 180, 0), width=1.8)
d.arrow(e, a, exit=(0.5, 1), entry=(138 / 180, 0), width=1.8)
d.arrow(a, s, exit=(0.5, 1), entry=(0.5, 0), width=1.8)
for k, both in [(0, False), (1, True)]:
    c = cd[k]
    d.box(c + 16, Y2 + 10, 66, 116, "Neural<br>Tower", "neural", size=S, fill="#E4EEF8")
    d.box(c + w - 82, Y2 + 10, 66, 116, "Env.<br>Tower", "env", size=S, fill="#EAF3E2")
    d.arrow(pts=[(c + 84, Y2 + 50), (c + w - 84, Y2 + 50)], width=2)
    if both:
        d.arrow(pts=[(c + w - 84, Y2 + 90), (c + 84, Y2 + 90)], width=2)

# --- second row: the prior's environmental content
ys = (1036, 1070, 1130, 1260, 1302)
row_heads(*ys[1:])
w = 207.5
cp = group(156, ys, "ISOLATING THE PRIOR (PECM, ORIGINAL DESIGN)", w,
           ["<b>Pure-Noise<br>Input</b>", "<b>Matched<br>Random Input</b>", "<b>Removed<br>at Test</b>",
            "<b>Prior<br>Ablations</b>"],
           ["Focus: Whether the prior's effect needs environmental content, or only an input tied to the label"])
Y2 = ys[2]
inputs = [("Random vectors,<br>no environmental<br>data", True),
          ("Synthetic vectors<br>with the prior's class<br>means, covariances", True),
          ("Training: prior<br>Test: grid mean", False),
          ("Reversed prior ·<br>random pairing ·<br>shuffled labels", False)]
for k, (lab, dashed) in enumerate(inputs):
    c = cp[k]
    i = d.box(c + 14, Y2 + 8, w - 28, 64, lab, "link", size=S, fill="#EFE8F6", dashed=dashed)
    e = d.box(c + 20, Y2 + 90, w - 40, 34, "Environmental Tower", "env", size=S, fill="#EAF3E2")
    d.arrow(i, e, exit=(0.5, 1), entry=(0.5, 0), width=1.8)

d.save("fig5_protocol.drawio")
print("wrote fig5_protocol.drawio")
