"""Figure 4 (Study 1 protocol). Part A: subject-level five-fold cross-validation, repeated three
times, from the study population to per-participant performance. Part B: the comparison framework
(alignment, fusion operator, and the controls that isolate the prior).

Drawn on a 960 px canvas so that, scaled to the text width, the smallest label prints at about
6 pt (the same scale as Figures 1 and 3)."""
from drawio_lib import Diagram

W, H = 960, 644
d = Diagram("Study 1 protocol", W, H)
TRAIN, VAL, TEST, FOLD = "#A9C8EA", "#CBB8E3", "#F6D36B", "#E8E8E8"
HEAD, MUTED = "#EDEDED", "#444444"
BAR, T, B = 17, 16, 15                       # section bars, titles, body (nothing smaller than 15)


def bar(y, label):
    d.box(8, y, W - 16, 30, f"<b>{label}</b>", "grey", size=BAR, fill="#E3E3E3", rounded=False)


def line(p, q, color="#333333", w=2):
    d.arrow(pts=[p, q], end="none", color=color, width=w)


# ============================== PART A
d.box(4, 4, W - 8, 308, "", "grey", fill="#FFFFFF", rounded=False)
bar(4, "PART A: SUBJECT-LEVEL FIVE-FOLD CROSS-VALIDATION, REPEATED THREE TIMES")
RY, RS, RH = 90, 36, 30                      # fold-grid rows
MID = RY + 2 * RS + RH / 2

# --- study population
d.text(12, 38, 200, 46, "<b>STUDY POPULATION</b><br>(N = 42)", size=T)
d.box(16, 88, 192, 48, "EEG: 30 channels,<br>400 epochs each", "neural", size=B, fill="#E4EEF8")
d.box(16, 142, 192, 48, "Astana grid:<br>100 cells × 4 layers", "env", size=B, fill="#EAF3E2")
d.box(16, 196, 192, 48, "5 affective states,<br>80 epochs each", "out", size=B, fill="#FCEEE2")
d.text(12, 248, 200, 58, "30 for selection;<br>lockbox of 12<br>scored once", size=B, color=MUTED)

# --- cross-validation loop
d.text(236, 38, 330, 24, "<b>CROSS-VALIDATION LOOP</b>", size=T)
d.text(248, 62, 204, 24, "training folds", size=B)
d.text(454, 62, 48, 24, "val.", size=B)
d.text(504, 62, 52, 24, "test", size=B)
line((240, RY + RH / 2), (240, RY + 4 * RS + RH / 2))
d.arrow(pts=[(210, MID), (240, MID)], width=2.5, end="none")
for k in range(5):
    y = RY + k * RS
    d.arrow(pts=[(240, y + RH / 2), (246, y + RH / 2)], width=2)
    for c, f in enumerate([f for f in range(1, 6) if f != k + 1]):
        d.box(248 + c * 52, y, 48, RH, str(f), "grey", size=B, fill=TRAIN, stroke_w=1)
    d.box(458, y, 40, RH, "val", "grey", size=B, fill=VAL, stroke_w=1)
    d.box(504, y, 52, RH, str(k + 1), "grey", size=B, fill=TEST, stroke_w=1)
    line((556, y + RH / 2), (566, y + RH / 2))
line((566, RY + RH / 2), (566, RY + 4 * RS + RH / 2))
d.arrow(pts=[(566, MID), (584, MID)], width=2.5)
d.text(236, 268, 330, 40, "val.: 4 participants of the training<br>folds, for early stopping and tuning",
       size=B, color=MUTED)

# --- model training
d.text(584, 38, 136, 46, "<b>MODEL<br>TRAINING</b>", size=T)
d.box(586, 88, 132, 176, "", "grey", fill="#F0F0F0", stroke_w=1.5)
d.box(596, 98, 112, 40, "EEG tower", "neural", size=B, fill="#E4EEF8")
d.text(596, 140, 112, 22, "<b>+</b>", size=20)
d.box(596, 164, 112, 40, "Graph tower", "env", size=B, fill="#EAF3E2")
d.text(590, 208, 124, 52, "PECM only in<br>training folds", size=B, color=MUTED)
d.arrow(pts=[(718, 114), (744, 114)], width=2.5)

# --- test and performance
d.text(744, 38, 204, 46, "<b>TEST AND<br>PERFORMANCE</b>", size=T)
d.box(746, 88, 200, 52, "<b>Test fold</b><br>unseen participants", "grey", size=B, fill=TEST, stroke_w=2)
d.text(744, 144, 204, 44, "environmental input<br>drawn without the label", size=B, color=MUTED)
d.box(746, 192, 200, 70, "per held-out participant,<br>mean of 3 repetitions,<br>95% bootstrap CI", "grey",
      size=B, fill="#FFFFFF", stroke_w=1.5)

# --- legend
for i, (lab, col) in enumerate([("training", TRAIN), ("validation", VAL), ("test", TEST)]):
    x = 596 + i * 118
    d.box(x, 278, 20, 18, "", "grey", fill=col, stroke_w=1, rounded=False)
    d.text(x + 24, 274, 92, 26, lab, size=B, align="left")

# ============================== PART B
YB = 320
d.box(4, YB, W - 8, 320, "", "grey", fill="#FFFFFF", rounded=False)
bar(YB, "PART B: COMPARISON FRAMEWORK AND ABLATION CONDITIONS")
Y0, Y1, Y2, Y3, Y4 = 356, 400, 446, 584, 634          # group title, configuration, components, focus, end

groups = [  # (title, sub-column width, configuration labels, focus)
    ("ISOLATING ALIGNMENT<br>STRATEGY", 116,
     ["<b>Baseline</b><br>EEG only", "<b>Multimodal</b><br>label-free"],
     "Focus: does an input drawn<br>without the label add to EEG?"),
    ("ISOLATING THE<br>FUSION OPERATOR", 124,
     ["<b>One vector</b><br>per tower", "<b>Token-level</b><br>attention"],
     "Focus: does any fusion operator<br>change what EEG alone gives?"),
    ("ISOLATING THE PRIOR (PECM)", 108,
     ["<b>Pure noise</b>", "<b>Matched<br>random</b>", "<b>Removed<br>at test</b>", "<b>Prior<br>ablations</b>"],
     "Focus: does the prior's effect need environmental<br>content, or only an input tied to the label?"),
]
x = 12
cells = []
for title, w, configs, focus in groups:
    gw = w * len(configs)
    d.box(x, Y0, gw, Y1 - Y0, f"<b>{title}</b>", "grey", size=B, fill=HEAD, stroke_w=1, rounded=False)
    for j, cfg in enumerate(configs):
        cx = x + j * w
        d.box(cx, Y1, w, Y2 - Y1, cfg, "grey", size=B, fill="#FFFFFF", stroke_w=1, rounded=False)
        d.box(cx, Y2, w, Y3 - Y2, "", "grey", fill="#FFFFFF", stroke_w=1, rounded=False)
        cells.append((cx, w))
    d.box(x, Y3, gw, Y4 - Y3, focus, "grey", size=B, fill="#FFFFFF", stroke_w=1, rounded=False)
    x += gw + 8

# alignment strategy
c, w = cells[0]
n = d.box(c + 22, Y2 + 16, 72, 36, "EEG", "neural", size=B, fill="#E4EEF8")
s = d.box(c + 14, Y2 + 94, 88, 32, "Softmax", "out", size=B, fill="#FCE8B8")
d.arrow(n, s, exit=(0.5, 1), entry=(0.5, 0), width=2)
c, w = cells[1]
n = d.box(c + 8, Y2 + 10, 48, 32, "EEG", "neural", size=B, fill="#E4EEF8")
e = d.box(c + 60, Y2 + 10, 48, 32, "Env.", "env", size=B, fill="#EAF3E2")
a = d.box(c + 10, Y2 + 58, 96, 28, "Attention", "out", size=B, fill="#F8D3C3")
s = d.box(c + 16, Y2 + 104, 84, 28, "Softmax", "out", size=B, fill="#FCE8B8")
d.arrow(n, a, exit=(0.5, 1), entry=(24 / 96, 0), width=1.8)
d.arrow(e, a, exit=(0.5, 1), entry=(72 / 96, 0), width=1.8)
d.arrow(a, s, exit=(0.5, 1), entry=(0.5, 0), width=1.8)

# fusion operator: one vector per tower (single-key attention in one or both directions, concatenation)
# against token-level attention between the 31 EEG tokens and the 100 grid-node tokens
c, w = cells[2]
d.box(c + 22, Y2 + 14, 80, 32, "EEG", "neural", size=B, fill="#E4EEF8")
d.box(c + 22, Y2 + 94, 80, 32, "Env.", "env", size=B, fill="#EAF3E2")
d.arrow(pts=[(c + 48, Y2 + 47), (c + 48, Y2 + 92)], width=2)
d.arrow(pts=[(c + 76, Y2 + 93), (c + 76, Y2 + 48)], width=2)
c, w = cells[3]
xs = [c + 16 + i * 25 for i in range(4)]
for x in xs:
    d.box(x, Y2 + 14, 17, 30, "", "neural", fill="#E4EEF8", stroke_w=1, rounded=False)
    d.box(x, Y2 + 96, 17, 30, "", "env", fill="#EAF3E2", stroke_w=1, rounded=False)
for x1 in xs:
    for x2 in xs:
        d.arrow(pts=[(x1 + 8.5, Y2 + 45), (x2 + 8.5, Y2 + 95)], end="none", color="#8A8A8A", width=1, straight=True)

# controls that isolate the prior
inputs = [("Random<br>vectors, no<br>env. data", True),
          ("Class means,<br>covariances;<br>no env. data", True),
          ("Train: prior<br>Test: grid<br>mean", False),
          ("Reversed,<br>label-free,<br>shuffled", False)]
for k, (lab, dashed) in enumerate(inputs):
    c, w = cells[4 + k]
    i = d.box(c + 4, Y2 + 8, w - 8, 66, lab, "link", size=B, fill="#EFE8F6", dashed=dashed)
    e = d.box(c + 12, Y2 + 96, w - 24, 30, "Env. tower", "env", size=B, fill="#EAF3E2")
    d.arrow(i, e, exit=(0.5, 1), entry=(0.5, 0), width=1.8)

d.save("fig5_protocol.drawio")
print("wrote fig5_protocol.drawio")
