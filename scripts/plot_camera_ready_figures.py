"""Camera-ready Figures A (evidence map) and B (resplit results) for NeurIPS 2026.

Reads ONLY the audited five-resplit artifacts listed in E2E_AUDIT.json (results/ in the
release, staged under runs/rebuttal/ as described in the README):
  runs/rebuttal/reviewer3_controls/unconditional_selected_all_seeds.csv
  runs/rebuttal/reviewer3_controls/raw_control_all_seeds.csv
  runs/rebuttal/reviewer3_controls/conditional_all_seeds.csv
  runs/rebuttal/reviewer3_controls/layer_permuted_selected_all_seeds.csv
  runs/rebuttal/reviewer_replications_5seed/paired_patch_per_resplit.csv
  runs/rebuttal/reviewer_replications_5seed/moment_per_resplit.csv
  runs/rebuttal/reviewer_replications_5seed/moment_random_structural_per_resplit.csv
  runs/rebuttal/reviewer_replications_5seed/moment_dynamic_per_resplit.csv
  runs/rebuttal/reviewer_replications_5seed/moment_interchange_per_resplit.csv
  runs/rebuttal/reviewer_replications_5seed/transfer_per_resplit.csv

Writes:
  paper/neurips2026/figures/fig_evidence_map.pdf (+ .png at 200 dpi)
  paper/neurips2026/figures/fig_resplit_results.pdf (+ .png at 200 dpi)
  paper/neurips2026/figures/cr_figure_values.csv

Deterministic: no RNG anywhere (fixed positional offsets for dots).
A win = a resplit with a strictly positive paired difference.
Half-widths = Student-t(4) two-sided 95% across the five resplits.
"""

import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats as sstats

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "runs", "rebuttal", "reviewer3_controls")
R5 = os.path.join(ROOT, "runs", "rebuttal", "reviewer_replications_5seed")
FIGDIR = os.path.join(ROOT, "paper", "neurips2026", "figures")

SEEDS = ["42", "43", "44", "45", "46"]
T95 = float(sstats.t.ppf(0.975, 4))

TAX_LABELS = ["frequency_bucket", "metric_type", "domain", "cardinality_bucket"]
PRETTY = {
    "frequency_bucket": "Cadence",
    "metric_type": "Metric type",
    "domain": "Domain",
    "cardinality_bucket": "Cardinality",
    "future_burstiness": "Future burstiness",
    "coordination": "Coordination",
}
# conditional_key -> taxonomy label it stands in for
COND2LABEL = {
    "frequency_given_infra_gauge": "frequency_bucket",
    "metric_type_given_app_short": "metric_type",
    "domain_given_gauge_short": "domain",
}
BLENDS = ["0.25", "0.5", "1.0"]

# ---------------------------------------------------------------- style
# Fig A diverging fills by k.
FILL = {0: "#7E2F24", 1: "#C96F3B", 2: "#E2E2E2",
        3: "#A3A3A3", 4: "#3F88C5", 5: "#1D4E7A"}
INK = {0: "#FFFFFF", 1: "#1A1A1A", 2: "#1A1A1A",
       3: "#1A1A1A", 4: "#1A1A1A", 5: "#FFFFFF"}
TOTO_BLUE = "#205A94"    # Fig B Toto series (marker shape + label carry identity too)
MOMENT_RED = "#7E2F24"   # Fig B MOMENT series
GREY_DOT = "#4A4A4A"
AXIS_GREY = "#6E6E6E"
# "Not tested" cells only (data-cell FILL/INK unchanged): slightly darker and
# thicker hatch for ~110 dpi print legibility; en-dash darker + larger but
# still clearly empty (lighter than the darkest data-cell text #1A1A1A).
NOTEST_EDGE = "#757575"
NOTEST_EDGE_LW = 0.7
NOTEST_HATCH = "////"
NOTEST_HATCH_LW = 0.7
NOTEST_DASH_COLOR = "#232323"
NOTEST_DASH_FS_MATRIX = 9.0
NOTEST_DASH_FS_KEY = 8.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "hatch.linewidth": NOTEST_HATCH_LW,
})

VALUES = []  # rows for cr_figure_values.csv


def add_val(figure, panel, label, endpoint, seed, value, source_file, columns, filt):
    VALUES.append({
        "figure": figure, "panel": panel, "label": label,
        "control_endpoint": endpoint, "seed": seed,
        "value": ("%.6f" % value) if isinstance(value, float) else str(value),
        "source_file": source_file, "source_columns": columns, "filter": filt,
    })


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_hw(xs):
    xs = list(xs)
    m = statistics.mean(xs)
    hw = T95 * statistics.stdev(xs) / (len(xs) ** 0.5)
    return m, hw


# ================================================================ data
un = load(os.path.join(R3, "unconditional_selected_all_seeds.csv"))
pre = {(r["label"], r["seed"]): float(r["test_macro_f1"])
       for r in un if r["weight_source"] == "pretrained"}
rnd = {(r["label"], r["seed"]): float(r["test_macro_f1"])
       for r in un if r["weight_source"] == "random_init"}
rawbase = {(r["label"], r["seed"]): float(r["baseline_test_macro_f1"])
           for r in un if r["weight_source"] == "pretrained"}

rc = load(os.path.join(R3, "raw_control_all_seeds.csv"))
fam_scores_val = {}
fam_scores_test = {}
for r in rc:
    fam_scores_val.setdefault((r["label"], r["raw_window_family"]), []).append(
        float(r["val_macro_f1"]))
    fam_scores_test.setdefault((r["label"], r["raw_window_family"]), []).append(
        float(r["test_macro_f1"]))
STRONG = {}  # label -> family with highest mean VALIDATION macro-F1
STRONG_TEST = {}  # same rule on test macro-F1 (must agree; assertion below)
for lab in TAX_LABELS:
    fams = sorted({f for (l, f) in fam_scores_val if l == lab})
    STRONG[lab] = max(
        fams, key=lambda f: statistics.mean(fam_scores_val[(lab, f)]))
    STRONG_TEST[lab] = max(
        fams, key=lambda f: statistics.mean(fam_scores_test[(lab, f)]))
_EXPECTED = {
    "frequency_bucket": "fno",
    "metric_type": "gbdt_spectral_autocorrelation_last_patch",
    "domain": "gbdt_spectral_autocorrelation_last_patch",
    "cardinality_bucket": "fno",
}
for _lab in TAX_LABELS:
    assert STRONG[_lab] == _EXPECTED[_lab], (
        "validation rule selected %s for %s, expected %s"
        % (STRONG[_lab], _lab, _EXPECTED[_lab]))
    assert STRONG_TEST[_lab] == _EXPECTED[_lab], (
        "test rule selected %s for %s, expected %s"
        % (STRONG_TEST[_lab], _lab, _EXPECTED[_lab]))
    assert STRONG[_lab] == STRONG_TEST[_lab], (
        "validation/test rules disagree for %s" % _lab)
fam_test = {(r["label"], r["raw_window_family"], r["seed"]): float(r["test_macro_f1"])
            for r in rc}
FAM_SHORT = {"fno": "FNO", "cnn": "CNN", "transformer": "Patch Transformer",
             "gbdt_spectral_autocorrelation_last_patch": "GBDT"}

co = load(os.path.join(R3, "conditional_all_seeds.csv"))
cond_toto, cond_ri, cond_raw = {}, {}, {}
for r in co:
    key = r["conditional_key"]
    if r["weight_source"] == "pretrained":
        cond_toto[(key, r["seed"])] = float(r["test_macro_f1"])
        cond_raw[(key, r["seed"])] = float(r["baseline_test_macro_f1"])
    else:
        cond_ri[(key, r["seed"])] = float(r["test_macro_f1"])

lp = load(os.path.join(R3, "layer_permuted_selected_all_seeds.csv"))
permd = {(r["label"], r["seed"]): float(r["test_macro_f1"]) for r in lp}

pp = load(os.path.join(R5, "paired_patch_per_resplit.csv"))
pp_cell = {(r["split_seed"], r["blend"]): r for r in pp}

mp = load(os.path.join(R5, "moment_per_resplit.csv"))
mom_test = {(r["label"], r["split_seed"]): float(r["test_macro_f1"]) for r in mp}
mom_raw = {(r["label"], r["split_seed"]): float(r["raw_test_macro_f1"]) for r in mp}
mr = load(os.path.join(R5, "moment_random_structural_per_resplit.csv"))
mom_ri = {(r["label"], r["split_seed"]): float(r["random_test_score"]) for r in mr}

md = load(os.path.join(R5, "moment_dynamic_per_resplit.csv"))
mom_dyn = {(r["label"], r["split_seed"]): float(r["test_r2"]) for r in md}
mom_dyn_raw = {(r["label"], r["split_seed"]): float(r["raw_test_r2"]) for r in md}

mi = load(os.path.join(R5, "moment_interchange_per_resplit.csv"))
mi_cell = {(r["split_seed"], r["blend"]): r for r in mi}

tr = load(os.path.join(R5, "transfer_per_resplit.csv"))

UNCOND_SRC = "runs/rebuttal/reviewer3_controls/unconditional_selected_all_seeds.csv"
RAW_SRC = "runs/rebuttal/reviewer3_controls/raw_control_all_seeds.csv"
COND_SRC = "runs/rebuttal/reviewer3_controls/conditional_all_seeds.csv"
PERM_SRC = "runs/rebuttal/reviewer3_controls/layer_permuted_selected_all_seeds.csv"
PP_SRC = "runs/rebuttal/reviewer_replications_5seed/paired_patch_per_resplit.csv"
MP_SRC = "runs/rebuttal/reviewer_replications_5seed/moment_per_resplit.csv"
MR_SRC = "runs/rebuttal/reviewer_replications_5seed/moment_random_structural_per_resplit.csv"
MD_SRC = "runs/rebuttal/reviewer_replications_5seed/moment_dynamic_per_resplit.csv"
MI_SRC = "runs/rebuttal/reviewer_replications_5seed/moment_interchange_per_resplit.csv"
TR_SRC = "runs/rebuttal/reviewer_replications_5seed/transfer_per_resplit.csv"


def wins(gaps):
    return sum(g > 0 for g in gaps)


# ------------------------------------------------- evidence-map cells
# cell[(row_label, col_key)] = k (0..5) or None for "not tested"
cell = {}
detail = {}  # (row, col) -> list of per-seed gap floats (for report stdout)

COLS = ["raw", "strongest", "strat", "randinit", "blockperm", "moment",
        "donor", "transfer"]
ROWS = ["frequency_bucket", "metric_type", "domain", "cardinality_bucket",
        "future_burstiness", "coordination"]

# Raw six-stat. probe column
for lab in TAX_LABELS:
    gaps = [pre[(lab, s)] - rawbase[(lab, s)] for s in SEEDS]
    cell[(lab, "raw")] = wins(gaps)
    detail[(lab, "raw")] = gaps
    for s, g in zip(SEEDS, gaps):
        add_val("A", "matrix", PRETTY[lab], "Raw six-stat. probe", s, g,
                UNCOND_SRC, "test_macro_f1,baseline_test_macro_f1",
                "weight_source=pretrained;gap=test-baseline")
    add_val("A", "matrix", PRETTY[lab], "Raw six-stat. probe", "wins",
            wins(gaps), UNCOND_SRC, "test_macro_f1,baseline_test_macro_f1",
            "weight_source=pretrained;count(gap>0)")
fb_gaps = [float(pp_cell[(s, "0.25")]["probe_test_r2"])
           - float(pp_cell[(s, "0.25")]["probe_raw_test_r2"]) for s in SEEDS]
cell[("future_burstiness", "raw")] = wins(fb_gaps)
detail[("future_burstiness", "raw")] = fb_gaps
for s, g in zip(SEEDS, fb_gaps):
    add_val("A", "matrix", "Future burstiness", "Raw six-stat. probe", s, g,
            PP_SRC, "probe_test_r2,probe_raw_test_r2",
            "blend=0.25 (blend-invariant);gap=probe-raw")
add_val("A", "matrix", "Future burstiness", "Raw six-stat. probe", "wins",
        wins(fb_gaps), PP_SRC, "probe_test_r2,probe_raw_test_r2",
        "blend=0.25;count(gap>0)")
# No audited Toto in-domain BOOM coordination R2 artifact -> not tested.
cell[("coordination", "raw")] = None

# Strongest raw-window model column
for lab in TAX_LABELS:
    fam = STRONG[lab]
    gaps = [pre[(lab, s)] - fam_test[(lab, fam, s)] for s in SEEDS]
    cell[(lab, "strongest")] = wins(gaps)
    detail[(lab, "strongest")] = gaps
    for s, g in zip(SEEDS, gaps):
        add_val("A", "matrix", PRETTY[lab],
                "Strongest raw-window model (%s)" % FAM_SHORT[fam], s, g,
                UNCOND_SRC + ";" + RAW_SRC,
                "test_macro_f1;val_macro_f1(family selection)",
                "family=%s selected by highest mean val_macro_f1;"
                "pretrained minus family,paired by seed" % fam)
    add_val("A", "matrix", PRETTY[lab],
            "Strongest raw-window model (%s)" % FAM_SHORT[fam], "wins",
            wins(gaps), UNCOND_SRC + ";" + RAW_SRC,
            "test_macro_f1;val_macro_f1(family selection)",
            "family=%s selected by highest mean val_macro_f1;"
            "count(gap>0)" % fam)
cell[("future_burstiness", "strongest")] = None
cell[("coordination", "strongest")] = None

# Common-support stratum column: min(wins vs RI, wins vs raw)
for key, lab in COND2LABEL.items():
    g_ri = [cond_toto[(key, s)] - cond_ri[(key, s)] for s in SEEDS]
    g_raw = [cond_toto[(key, s)] - cond_raw[(key, s)] for s in SEEDS]
    k = min(wins(g_ri), wins(g_raw))
    cell[(lab, "strat")] = k
    detail[(lab, "strat")] = (g_ri, g_raw)
    for s, g in zip(SEEDS, g_ri):
        add_val("A", "matrix", PRETTY[lab],
                "Common-support stratum vs random init", s, g,
                COND_SRC, "test_macro_f1",
                "conditional_key=%s;pretrained-random_init" % key)
    for s, g in zip(SEEDS, g_raw):
        add_val("A", "matrix", PRETTY[lab],
                "Common-support stratum vs raw six-stat", s, g,
                COND_SRC, "test_macro_f1,baseline_test_macro_f1",
                "conditional_key=%s;pretrained-baseline" % key)
    add_val("A", "matrix", PRETTY[lab], "Common-support stratum", "wins", k,
            COND_SRC, "test_macro_f1,baseline_test_macro_f1",
            "conditional_key=%s;min(vsRI,vsRaw)" % key)
cell[("cardinality_bucket", "strat")] = None
cell[("future_burstiness", "strat")] = None
cell[("coordination", "strat")] = None

# Random init. column
for lab in TAX_LABELS:
    gaps = [pre[(lab, s)] - rnd[(lab, s)] for s in SEEDS]
    cell[(lab, "randinit")] = wins(gaps)
    detail[(lab, "randinit")] = gaps
    for s, g in zip(SEEDS, gaps):
        add_val("A", "matrix", PRETTY[lab], "Random init.", s, g,
                UNCOND_SRC, "test_macro_f1",
                "pretrained-random_init,paired by seed")
    add_val("A", "matrix", PRETTY[lab], "Random init.", "wins", wins(gaps),
            UNCOND_SRC, "test_macro_f1", "count(gap>0)")
cell[("future_burstiness", "randinit")] = None
cell[("coordination", "randinit")] = None

# Block perm. column
for lab in TAX_LABELS:
    gaps = [pre[(lab, s)] - permd[(lab, s)] for s in SEEDS]
    cell[(lab, "blockperm")] = wins(gaps)
    detail[(lab, "blockperm")] = gaps
    for s, g in zip(SEEDS, gaps):
        add_val("A", "matrix", PRETTY[lab], "Block perm.", s, g,
                UNCOND_SRC + ";" + PERM_SRC, "test_macro_f1",
                "pretrained-layer_permuted,paired by seed")
    add_val("A", "matrix", PRETTY[lab], "Block perm.", "wins", wins(gaps),
            UNCOND_SRC + ";" + PERM_SRC, "test_macro_f1", "count(gap>0)")
cell[("future_burstiness", "blockperm")] = None
cell[("coordination", "blockperm")] = None

# MOMENT-base column
for lab in TAX_LABELS:
    g_ri = [mom_test[(lab, s)] - mom_ri[(lab, s)] for s in SEEDS]
    g_raw = [mom_test[(lab, s)] - mom_raw[(lab, s)] for s in SEEDS]
    k = min(wins(g_ri), wins(g_raw))
    cell[(lab, "moment")] = k
    detail[(lab, "moment")] = (g_ri, g_raw)
    for s, g in zip(SEEDS, g_ri):
        add_val("A", "matrix", PRETTY[lab],
                "MOMENT-base vs MOMENT random init", s, g,
                MP_SRC + ";" + MR_SRC, "test_macro_f1,random_test_score",
                "paired by seed")
    for s, g in zip(SEEDS, g_raw):
        add_val("A", "matrix", PRETTY[lab],
                "MOMENT-base vs raw six-stat", s, g,
                MP_SRC, "test_macro_f1,raw_test_macro_f1", "paired by seed")
    add_val("A", "matrix", PRETTY[lab], "MOMENT-base", "wins", k,
            MP_SRC + ";" + MR_SRC,
            "test_macro_f1,raw_test_macro_f1,random_test_score",
            "min(vsRI,vsRaw)")
for lab in ["future_burstiness", "coordination"]:
    gaps = [mom_dyn[(lab, s)] - mom_dyn_raw[(lab, s)] for s in SEEDS]
    cell[(lab, "moment")] = wins(gaps)
    detail[(lab, "moment")] = gaps
    for s, g in zip(SEEDS, gaps):
        add_val("A", "matrix", PRETTY[lab], "MOMENT-base vs raw six-stat",
                s, g, MD_SRC, "test_r2,raw_test_r2", "paired by seed")
    add_val("A", "matrix", PRETTY[lab], "MOMENT-base vs raw six-stat",
            "wins", wins(gaps), MD_SRC, "test_r2,raw_test_r2",
            "count(gap>0)")

# Donor exchange column: max over blends of resplits with burst_win_fraction>0.5
donor_per_blend = {}
for b in BLENDS:
    cnt = sum(float(pp_cell[(s, b)]["burst_win_fraction"]) > 0.5 for s in SEEDS)
    donor_per_blend[b] = cnt
    for s in SEEDS:
        add_val("A", "matrix", "Future burstiness",
                "Donor exchange burst_win_fraction b=%s" % b, s,
                float(pp_cell[(s, b)]["burst_win_fraction"]),
                PP_SRC, "burst_win_fraction", "none")
cell[("future_burstiness", "donor")] = max(donor_per_blend.values())
add_val("A", "matrix", "Future burstiness", "Donor exchange", "wins",
        max(donor_per_blend.values()), PP_SRC, "burst_win_fraction",
        "max over blends of count(fraction>0.5)")
for lab in TAX_LABELS + ["coordination"]:
    cell[(lab, "donor")] = None

# External transfer column: max over FEV / LSTF of resplits with R2>0
trans_counts = {}
for b in ["fev", "lsf"]:
    vals = [float(r["transfer_r2_macro"]) for r in tr
            if r["label"] == "coordination" and r["benchmark"] == b]
    trans_counts[b] = (vals, sum(v > 0 for v in vals))
    for s, v in zip(SEEDS, vals):
        add_val("A", "matrix", "Coordination",
                "External transfer R2 benchmark=%s" % b, s, v,
                TR_SRC, "transfer_r2_macro",
                "label=coordination,benchmark=%s" % b)
cell[("coordination", "transfer")] = max(c for _, c in trans_counts.values())
add_val("A", "matrix", "Coordination", "External transfer", "wins",
        max(c for _, c in trans_counts.values()), TR_SRC, "transfer_r2_macro",
        "max over benchmarks of count(R2>0)")
for lab in TAX_LABELS + ["future_burstiness"]:
    cell[(lab, "transfer")] = None

# ------------------------------------------------- Figure B values
# Panel (a): Toto minus strongest-family test macro-F1, per resplit.
PANEL_A = {}
for lab in TAX_LABELS:
    fam = STRONG[lab]
    gaps = [pre[(lab, s)] - fam_test[(lab, fam, s)] for s in SEEDS]
    m, hw = mean_hw(gaps)
    PANEL_A[lab] = {"fam": fam, "gaps": gaps, "mean": m, "hw": hw,
                    "wins": wins(gaps)}
    for s, g in zip(SEEDS, gaps):
        add_val("B", "a", PRETTY[lab],
                "Toto minus %s" % FAM_SHORT[fam], s, g,
                UNCOND_SRC + ";" + RAW_SRC,
                "test_macro_f1;val_macro_f1(family selection)",
                "family=%s selected by highest mean val_macro_f1;"
                "paired by seed" % fam)
    add_val("B", "a", PRETTY[lab], "Toto minus %s" % FAM_SHORT[fam],
            "mean", m, UNCOND_SRC + ";" + RAW_SRC,
            "test_macro_f1;val_macro_f1(family selection)",
            "family=%s selected by highest mean val_macro_f1;"
            "mean over 5 resplits" % fam)
    add_val("B", "a", PRETTY[lab], "Toto minus %s" % FAM_SHORT[fam],
            "halfwidth", hw, UNCOND_SRC + ";" + RAW_SRC,
            "test_macro_f1;val_macro_f1(family selection)",
            "family=%s selected by highest mean val_macro_f1;"
            "Student-t(4) 95%% half-width" % fam)
    add_val("B", "a", PRETTY[lab], "Toto minus %s" % FAM_SHORT[fam],
            "wins", wins(gaps), UNCOND_SRC + ";" + RAW_SRC,
            "test_macro_f1;val_macro_f1(family selection)",
            "family=%s selected by highest mean val_macro_f1;"
            "count(gap>0)" % fam)

# Panel (b): probe check + forecast endpoint fractions per resplit.
# Blend invariance of probe_win_fraction is asserted, not assumed.
for s in SEEDS:
    toto_pwf = {pp_cell[(s, b)]["probe_win_fraction"] for b in BLENDS}
    assert len(toto_pwf) == 1, "Toto probe_win_fraction varies by blend"
    mom_pwf = {mi_cell[(s, b)]["probe_win_fraction"] for b in BLENDS}
    assert len(mom_pwf) == 1, "MOMENT probe_win_fraction varies by blend"

TOTO_PROBE = [float(pp_cell[(s, "0.25")]["probe_win_fraction"]) for s in SEEDS]
MOM_PROBE = [float(mi_cell[(s, "0.25")]["probe_win_fraction"]) for s in SEEDS]
TOTO_FORE = {b: [float(pp_cell[(s, b)]["burst_win_fraction"]) for s in SEEDS]
             for b in BLENDS}
MOM_FORE = {b: [float(mi_cell[(s, b)]["future_mae_real_better_fraction"])
                for s in SEEDS] for b in BLENDS}

PANEL_B = {"Toto": {"Probe check": TOTO_PROBE},
           "MOMENT": {"Probe check": MOM_PROBE}}
for b in BLENDS:
    PANEL_B["Toto"]["beta %s" % b] = TOTO_FORE[b]
    PANEL_B["MOMENT"]["beta %s" % b] = MOM_FORE[b]
PANEL_B_SRC = {
    ("Toto", "Probe check"): (PP_SRC, "probe_win_fraction", "blend=0.25"),
    ("MOMENT", "Probe check"): (MI_SRC, "probe_win_fraction", "blend=0.25"),
}
for b in BLENDS:
    PANEL_B_SRC[("Toto", "beta %s" % b)] = (PP_SRC, "burst_win_fraction",
                                            "blend=%s" % b)
    PANEL_B_SRC[("MOMENT", "beta %s" % b)] = (
        MI_SRC, "future_mae_real_better_fraction", "blend=%s" % b)
for series, cols in PANEL_B.items():
    for colname, vals in cols.items():
        src, col, filt = PANEL_B_SRC[(series, colname)]
        for s, v in zip(SEEDS, vals):
            add_val("B", "b", colname, series, s, v, src, col, filt)
        m, hw = mean_hw(vals)
        add_val("B", "b", colname, series, "mean", m, src, col,
                filt + ";mean over 5 resplits")
        add_val("B", "b", colname, series, "halfwidth", hw, src, col,
                filt + ";Student-t(4) 95% half-width")


# ================================================================ Figure A
# One-tier group headers, each spanning exactly its own columns.
COL_SHORT = [[u"Six-stat.", u"probe"], [u"Strongest", u"raw-window"],
             [u"Common-", u"support"], [u"Random", u"init."],
             [u"Block", u"perm."], [u"MOMENT-", u"base"],
             [u"Donor", u"exchange"], [u"External", u"transfer"]]
GROUPS = [("Readout", 0, 2), ("Trained config.", 3, 4), ("Cross-model", 5, 5),
          ("Behavior", 6, 6), ("Transfer", 7, 7)]


def plot_evidence_map():
    fig = plt.figure(figsize=(5.5, 2.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 2.0)
    ax.axis("off")

    # geometry (inches in figure coords)
    lab_w = 0.95          # row-label column (narrowed to give matrix room)
    mat_x0 = lab_w + 0.27  # +0.27 centers the block: was 0.02 L / 0.57 R
    mat_w = 3.97          # matrix width
    cw = mat_w / 8.0
    gap_h = 0.10          # visual gap between row blocks
    row_h = 0.175
    key_y0 = 0.02
    key_h = 0.28
    mat_y0 = key_y0 + key_h + 0.06
    # rows top->bottom in data order; y of row i (0-indexed) top edge:
    nrows = 6
    mat_h = nrows * row_h + gap_h
    head_h = 2.15 - (mat_y0 + mat_h)  # header band height

    def row_top(i):
        # i = 0..5 top->bottom; extra gap after row 3
        extra = gap_h if i >= 4 else 0.0
        return mat_y0 + mat_h - i * row_h - extra - row_h

    # matrix backing so white gaps read as gaps
    ax.add_patch(Rectangle((mat_x0 - 0.012, mat_y0 - 0.012), mat_w + 0.024,
                           mat_h + 0.024, facecolor="#EFEFEF", edgecolor="none",
                           zorder=0))

    for i, lab in enumerate(ROWS):
        yt = row_top(i)
        # row label
        ax.text(mat_x0 - 0.05, yt + row_h / 2.0, PRETTY[lab], ha="right",
                va="center", fontsize=7.5)
        for j, ckey in enumerate(COLS):
            x = mat_x0 + j * cw
            k = cell[(lab, ckey)]
            rect = Rectangle((x + 0.008, yt + 0.008), cw - 0.016, row_h - 0.016,
                             zorder=2)
            if k is None:
                rect.set_facecolor("white")
                rect.set_edgecolor(NOTEST_EDGE)
                rect.set_linewidth(NOTEST_EDGE_LW)
                rect.set_hatch(NOTEST_HATCH)
                ax.add_patch(rect)
                ax.text(x + cw / 2.0, yt + row_h / 2.0, "\u2013",
                        ha="center", va="center",
                        fontsize=NOTEST_DASH_FS_MATRIX,
                        color=NOTEST_DASH_COLOR, zorder=3)
            else:
                rect.set_facecolor(FILL[k])
                rect.set_edgecolor("white")
                rect.set_linewidth(0.9)
                ax.add_patch(rect)
                ax.text(x + cw / 2.0, yt + row_h / 2.0, "%d/5" % k,
                        ha="center", va="center", fontsize=7.5,
                        color=INK[k], zorder=3)

    # Row-group tags at the far left (kept: 2x-crop inspection shows
    # they clear all row labels).
    tax_mid = (row_top(0) + row_h / 2.0 + row_top(3) + row_h / 2.0) / 2.0
    dyn_mid = (row_top(4) + row_h / 2.0 + row_top(5) + row_h / 2.0) / 2.0
    ax.text(0.34, tax_mid, "Taxonomy", ha="center", va="center",
            fontsize=7.0, color="#555555", rotation=90)
    ax.text(0.34, dyn_mid, "Dynamic", ha="center", va="center",
            fontsize=7.0, color="#555555", rotation=90)
    # Column headers: horizontal labels centered over each column, plus one
    # tier of group headers, each with its own underline (small gap between
    # adjacent underlines via the 0.02 inset on each side).
    hy = mat_y0 + mat_h
    for j, lines in enumerate(COL_SHORT):
        x = mat_x0 + j * cw + cw / 2.0
        if len(lines) == 2:
            ax.text(x, hy + 0.175, lines[0], ha="center", va="center",
                    fontsize=7.0)
            ax.text(x, hy + 0.075, lines[1], ha="center", va="center",
                    fontsize=7.0)
        else:
            ax.text(x, hy + 0.125, lines[0], ha="center", va="center",
                    fontsize=7.0)
    rule_y = hy + 0.27
    for gtxt, j0, j1 in GROUPS:
        x0 = mat_x0 + j0 * cw
        x1 = mat_x0 + (j1 + 1) * cw
        ax.text((x0 + x1) / 2.0, rule_y + 0.03, gtxt, ha="center",
                va="bottom", fontsize=7.0)
        ax.plot([x0 + 0.02, x1 - 0.02], [rule_y, rule_y],
                color=AXIS_GREY, lw=0.6, clip_on=False)

    # compact key below the matrix
    kx = mat_x0
    ax.text(kx, key_y0 + key_h - 0.02,
            "Cell: resplits won (k/5) by Toto (MOMENT in MOMENT-base column).",
            ha="left", va="top", fontsize=7.0, color="#333333")
    sw = 0.16
    y_sw = key_y0 + 0.06
    for i, k in enumerate([5, 4, 3, 2, 1, 0]):
        x = kx + i * 0.52
        ax.add_patch(Rectangle((x, y_sw), sw, 0.11, facecolor=FILL[k],
                               edgecolor="#6E6E6E", linewidth=0.4))
        ax.text(x + sw / 2.0, y_sw + 0.055, "%d/5" % k, ha="center",
                va="center", fontsize=7.0, color=INK[k])
    x = kx + 6 * 0.52
    ax.add_patch(Rectangle((x, y_sw), sw, 0.11, facecolor="white",
                           edgecolor=NOTEST_EDGE, linewidth=NOTEST_EDGE_LW,
                           hatch=NOTEST_HATCH))
    ax.text(x + sw / 2.0, y_sw + 0.055, "\u2013", ha="center", va="center",
            fontsize=NOTEST_DASH_FS_KEY, color=NOTEST_DASH_COLOR)
    ax.text(x + sw + 0.04, y_sw + 0.055, "not tested", ha="left",
            va="center", fontsize=7.0, color="#333333")
    return fig


# ================================================================ Figure B
DOT_OFF = [-0.20, -0.10, 0.0, 0.10, 0.20]


def _mean_err(ax, x, y, hw, color, marker, ms):
    ax.errorbar([x], [y], xerr=[hw], color=color, ecolor=color, elinewidth=0.9,
                capsize=2.2, capthick=0.9, marker=marker, markersize=ms,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5, lw=0)


def plot_resplit_results():
    # Height 1.9 (was 2.2): the legend now lives inside panel (b), so the
    # bottom ~0.3 in was empty. Axes keep their absolute size
    # (0.672*1.9 = 1.277 in = old 0.58*2.2); only figure-edge margin removed.
    fig = plt.figure(figsize=(5.5, 1.9))

    # ---- panel (a): broken x-axis, shared y.
    # The right segment is extended to 0.45 (past the last tick at 0.2) so
    # the per-row family+wins annotations sit inside the axes, on the scale.
    axL = fig.add_axes([0.115, 0.186, 0.21, 0.672])
    axR = fig.add_axes([0.33, 0.186, 0.24, 0.672])
    ypos = {"frequency_bucket": 3, "metric_type": 2, "domain": 1,
            "cardinality_bucket": 0}
    for lab in TAX_LABELS:
        d = PANEL_A[lab]
        y = ypos[lab]
        ax = axL if lab == "cardinality_bucket" else axR
        for k, g in enumerate(d["gaps"]):
            ax.plot(g, y + DOT_OFF[k], marker="o", markersize=2.6,
                    color=GREY_DOT, markeredgewidth=0, zorder=4)
        _mean_err(ax, d["mean"], y, d["hw"], TOTO_BLUE, "D", 4.2)
        if lab != "cardinality_bucket":
            ax.text(0.235, y, "%s \u00b7 %d/5" % (FAM_SHORT[d["fam"]],
                                                  d["wins"]),
                    ha="left", va="center", fontsize=7.0)
    axL.set_xlim(-0.60, -0.38)
    axR.set_xlim(-0.08, 0.45)
    for a in (axL, axR):
        a.set_ylim(-0.5, 3.5)
        a.set_yticks([3, 2, 1, 0])
        a.tick_params(labelsize=7.5, length=2.0, pad=2.0)
        a.grid(axis="x", color="#E3E3E3", linewidth=0.4, zorder=0)
        for spine in a.spines.values():
            spine.set_color(AXIS_GREY)
    axL.set_yticklabels(["Cadence", "Metric type", "Domain", "Cardinality"])
    axR.set_yticklabels([])
    axL.set_xticks([-0.6, -0.5, -0.4])
    axR.set_xticks([0.0, 0.1, 0.2])
    axR.axvline(0, color=AXIS_GREY, linestyle=(0, (3, 2)), linewidth=0.7,
                zorder=1)
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(left=False)
    # break marks on the facing spines
    d = 0.018
    kw = dict(color="black", linewidth=0.7, clip_on=False)
    axL.plot((1 - d, 1 + d), (-d, +d), transform=axL.transAxes, **kw)
    axL.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axL.transAxes, **kw)
    axR.plot((-d, +d), (-d, +d), transform=axR.transAxes, **kw)
    axR.plot((-d, +d), (1 - d, 1 + d), transform=axR.transAxes, **kw)
    # cardinality annotation (no room on the left segment: place above point)
    cd = PANEL_A["cardinality_bucket"]
    axL.text(cd["mean"], 0.38, "%s \u00b7 %d/5" % (FAM_SHORT[cd["fam"]],
                                                   cd["wins"]),
             ha="center", va="bottom", fontsize=7.0)
    fig.text(0.3425, 0.081,
             "Toto \u2212 strongest raw-window test macro-F1",
             ha="center", va="top", fontsize=7.5)
    axL.text(-0.06, 1.03, "(a)", transform=axL.transAxes, fontsize=8.0,
             va="bottom", ha="right")

    # ---- panel (b): readout vs forecast endpoint.
    # The y-axis label sits horizontally above the panel: a vertical label
    # would collide with panel (a)'s right-side annotations in the gutter.
    axB = fig.add_axes([0.64, 0.186, 0.335, 0.672])
    cats = ["Probe\ncheck", r"$\beta$ = 0.25", r"$\beta$ = 0.5",
            r"$\beta$ = 1.0"]
    colkeys = ["Probe check", "beta 0.25", "beta 0.5", "beta 1.0"]
    series = [("Toto", TOTO_BLUE, "o", -0.09), ("MOMENT", MOMENT_RED, "^", 0.09)]
    for name, color, marker, dx in series:
        for j, ck in enumerate(colkeys):
            vals = PANEL_B[name][ck]
            for k, v in enumerate(vals):
                axB.plot(j + dx, v + DOT_OFF[k] * 0.12, marker=marker,
                         markersize=2.6, color=color, alpha=0.55,
                         markeredgewidth=0, zorder=4)
            m, hw = mean_hw(vals)
            axB.errorbar([j + dx], [m], yerr=[hw], color=color, ecolor=color,
                         elinewidth=0.9, capsize=2.2, capthick=0.9,
                         marker=marker, markersize=4.2, markerfacecolor=color,
                         markeredgecolor="white", markeredgewidth=0.6,
                         zorder=5, lw=0)
    axB.set_xlim(-0.35, 3.35)
    axB.set_ylim(0.0, 1.0)
    axB.set_xticks(range(4))
    axB.set_xticklabels(cats, fontsize=7.5)
    axB.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axB.tick_params(labelsize=7.5, length=2.0, pad=2.0)
    axB.grid(axis="y", color="#E3E3E3", linewidth=0.4, zorder=0)
    for spine in axB.spines.values():
        spine.set_color(AXIS_GREY)
    axB.axhline(0.5, color=AXIS_GREY, linestyle=(0, (3, 2)), linewidth=0.7,
                zorder=1)
    axB.axvline(0.5, color="#9A9A9A", linewidth=0.6, zorder=1)
    axB.text(0.5, 1.10, "Fraction of 40 paired targets", ha="center",
             va="bottom", fontsize=7.5, transform=axB.transAxes)
    axB.text(-0.02, 1.03, "(b)", transform=axB.transAxes, fontsize=8.0,
             va="bottom", ha="right")

    # legend lives inside panel (b)'s empty upper-right region; entries
    # name each series' forecast endpoint.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TOTO_BLUE,
               markeredgecolor=TOTO_BLUE, markersize=4.5,
               label="Toto (forecast burstier)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=MOMENT_RED,
               markeredgecolor=MOMENT_RED, markersize=4.5,
               label="MOMENT (lower future MAE)"),
    ]
    axB.legend(handles=handles, loc="upper right", fontsize=7.0,
               frameon=True, facecolor="white", edgecolor="#9A9A9A",
               handletextpad=0.3, borderpad=0.4)
    return fig


# ================================================================ main
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    figA = plot_evidence_map()
    figA.savefig(os.path.join(FIGDIR, "fig_evidence_map.pdf"))
    figA.savefig(os.path.join(FIGDIR, "fig_evidence_map.png"), dpi=200)
    print("A size (in):", figA.get_size_inches())
    plt.close(figA)

    figB = plot_resplit_results()
    figB.savefig(os.path.join(FIGDIR, "fig_resplit_results.pdf"))
    figB.savefig(os.path.join(FIGDIR, "fig_resplit_results.png"), dpi=200)
    print("B size (in):", figB.get_size_inches())
    plt.close(figB)

    with open(os.path.join(FIGDIR, "cr_figure_values.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "panel", "label",
                                          "control_endpoint", "seed", "value",
                                          "source_file", "source_columns",
                                          "filter"])
        w.writeheader()
        w.writerows(VALUES)
    print("values rows:", len(VALUES))
    write_tikz_data()
    verify_tikz_data()

    print("\n== strongest family per label ==")
    for lab in TAX_LABELS:
        print(" ", PRETTY[lab], FAM_SHORT[STRONG[lab]])
    print("\n== evidence cells (k/5; . = not tested) ==")
    head = " " * 18 + "".join("%-9s" % c for c in COLS)
    print(head)
    for lab in ROWS:
        row = "%-18s" % PRETTY[lab]
        for c in COLS:
            k = cell[(lab, c)]
            row += "%-9s" % (". " if k is None else "%d/5" % k)
        print(row)
    print("\n== panel A ==")
    for lab in TAX_LABELS:
        d = PANEL_A[lab]
        print(" %s: mean=%.4f hw=%.4f wins=%d/5 gaps=%s"
              % (PRETTY[lab], d["mean"], d["hw"], d["wins"],
                 [round(g, 4) for g in d["gaps"]]))
    print("\n== panel B means ==")
    for name, cols in PANEL_B.items():
        for ck, vals in cols.items():
            m, hw = mean_hw(vals)
            print(" %s %s: mean=%.4f hw=%.4f n>0.5=%d/5" %
                  (name, ck, m, hw, sum(v > 0.5 for v in vals)))


# ================================================================ TikZ export (job K1)
# Writes data CSVs + generated TikZ snippets for the camera-ready TikZ
# figures (paper/neurips2026/figures/{data,tikz}/). Every value is asserted
# against cr_figure_values.csv (the audited values behind the matplotlib
# figures); failure raises AssertionError.
TIKZDIR = os.path.join(FIGDIR, "tikz")
DATADIR = os.path.join(FIGDIR, "data")

# Row/column order shared with figures/tikz/evidence_map.tex.
EV_ROWS = ROWS
EV_COLS = COLS

# Deterministic vertical jitter for interchange dots (same offsets as the
# matplotlib figure: DOT_OFF[k]*0.12 in data units).
INTERCHANGE_X = ["Probe check", "beta 0.25", "beta 0.5", "beta 1.0"]
XPOS = {"Probe check": 1, "beta 0.25": 2, "beta 0.5": 3, "beta 1.0": 4}


def _fmt6(x):
    return "%.6f" % x


def _cr_values(path):
    """cr_figure_values.csv keyed on (figure,panel,label,endpoint,seed)."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["figure"], r["panel"], r["label"], r["control_endpoint"],
                 r["seed"])] = r["value"]
    return out


def write_tikz_data():
    os.makedirs(TIKZDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)

    # ---- evidence map cells (k/5; None = not tested)
    with open(os.path.join(DATADIR, "evidence_map.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_key", "row_label", "col_key", "k"])
        for lab in EV_ROWS:
            for ckey in EV_COLS:
                k = cell[(lab, ckey)]
                w.writerow([lab, PRETTY[lab], ckey,
                            "" if k is None else k])
    with open(os.path.join(TIKZDIR, "evidence_map_cells.tex"), "w") as f:
        f.write("% Generated by scripts/plot_camera_ready_figures.py "
                "-- do not edit.\n")
        f.write("% \\evcell{row}{col}{k} rows/cols 0-based; k = 0..5 or X.\n")
        for i, lab in enumerate(EV_ROWS):
            for j, ckey in enumerate(EV_COLS):
                k = cell[(lab, ckey)]
                f.write("\\evcell{%d}{%d}{%s}\n"
                        % (i, j, "X" if k is None else k))

    # ---- resplit slopes: per-resplit Toto vs strongest-raw test macro-F1
    slope_rows = []
    for lab in TAX_LABELS:
        fam = STRONG[lab]
        for s in SEEDS:
            t = pre[(lab, s)]
            r = fam_test[(lab, fam, s)]
            slope_rows.append((lab, s, t, r, fam))
    with open(os.path.join(DATADIR, "resplit_slopes.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "label_pretty", "seed", "toto_f1", "raw_f1",
                    "raw_family", "raw_short", "toto_wins"])
        for lab, s, t, r, fam in slope_rows:
            w.writerow([lab, PRETTY[lab], s, _fmt6(t), _fmt6(r), fam,
                        FAM_SHORT[fam], 1 if t > r else 0])
    # generated per-panel \addplot lines (each file is \input inside one
    # axis env in figures/tikz/resplit_results.tex); coordinates at
    # 4 decimals. Wins macros live in resplit_slope_wins.tex.
    winnames = {"frequency_bucket": "Cadence", "metric_type": "MetricType",
                "domain": "Domain", "cardinality_bucket": "Cardinality"}
    with open(os.path.join(TIKZDIR, "resplit_slope_wins.tex"), "w") as f:
        f.write("% Generated by scripts/plot_camera_ready_figures.py "
                "-- do not edit.\n")
        for lab in TAX_LABELS:
            fam = STRONG[lab]
            f.write("\\newcommand{\\slopeWins%s}{%d/5}\n"
                    % (winnames[lab],
                       wins([pre[(lab, s)] - fam_test[(lab, fam, s)]
                             for s in SEEDS])))
    for lab in TAX_LABELS:
        fam = STRONG[lab]
        with open(os.path.join(TIKZDIR,
                               "resplit_slope_%s.tex" % winnames[lab]),
                  "w") as f:
            f.write("%% Generated -- do not edit. Panel %s (raw: %s).\n"
                    % (PRETTY[lab], FAM_SHORT[fam]))
            for s in SEEDS:
                t = pre[(lab, s)]
                r = fam_test[(lab, fam, s)]
                col = "crToto" if t > r else "crMoment"
                style = "crToto, thin, solid" if t > r else "crMoment, thin, dashed"
                f.write("\\addplot[%s] coordinates "
                        "{(0,%.4f) (1,%.4f)};\n" % (style, t, r))
                f.write("\\addplot[only marks, mark=*, mark size=1.5pt, "
                        "%s, mark options={draw=white, line width=0.3pt}] "
                        "coordinates {(0,%.4f) (1,%.4f)};\n" % (col, t, r))
            tm = statistics.mean(pre[(lab, s)] for s in SEEDS)
            rm = statistics.mean(fam_test[(lab, fam, s)] for s in SEEDS)
            f.write("\\addplot[only marks, mark=diamond*, mark size=3pt, "
                    "crToto, mark options={draw=white, line width=0.4pt}] "
                    "coordinates {(0,%.4f)};\n" % tm)
            f.write("\\addplot[only marks, mark=diamond*, mark size=3pt, "
                    "crMuted, mark options={draw=white, line width=0.4pt}] "
                    "coordinates {(1,%.4f)};\n" % rm)

    # ---- interchange dots + stats, one file pair per model
    for model in ["Toto", "MOMENT"]:
        dots, stats = [], []
        for xname in INTERCHANGE_X:
            vals = PANEL_B[model][xname]
            m, hw = mean_hw(vals)
            stats.append((xname, XPOS[xname], m, hw))
            for k, (s, v) in enumerate(zip(SEEDS, vals)):
                dots.append((xname, XPOS[xname], s, v,
                             v + DOT_OFF[k] * 0.12))
        tag = model.lower()
        with open(os.path.join(DATADIR, "interchange_%s_dots.csv" % tag),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "xpos", "seed", "value", "py"])
            for xname, xp, s, v, py in dots:
                w.writerow([xname, xp, s, _fmt6(v), _fmt6(py)])
        with open(os.path.join(DATADIR, "interchange_%s_stats.csv" % tag),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "xpos", "mean", "halfwidth"])
            for xname, xp, m, hw in stats:
                w.writerow([xname, xp, _fmt6(m), _fmt6(hw)])


def verify_tikz_data():
    """Every TikZ-plotted value must equal cr_figure_values.csv / sources."""
    cv = _cr_values(os.path.join(FIGDIR, "cr_figure_values.csv"))

    # (1) evidence cells: k == A/matrix wins rows in cr_figure_values.csv.
    with open(os.path.join(DATADIR, "evidence_map.csv"), newline="") as f:
        for r in csv.DictReader(f):
            lab, ckey = r["row_key"], r["col_key"]
            k = cell[(lab, ckey)]
            if k is None:
                assert r["k"] == "", (lab, ckey)
                continue
            assert r["k"] == str(k), (lab, ckey)
    ev_wins = {(l, c): cell[(l, c)] for l in ROWS for c in COLS
               if cell[(l, c)] is not None}
    assert len(ev_wins) == 28, len(ev_wins)  # 6+6+6+5+3+2 per F3 matrix
    # generated cells file round-trips the CSV.
    cells = {}
    with open(os.path.join(TIKZDIR, "evidence_map_cells.tex")) as f:
        for line in f:
            line = line.strip()
            if line.startswith("\\evcell"):
                p = line[len("\\evcell"):].replace("{",
                                                   " ").replace("}", " ")
                ri, ci, k = p.split()
                cells[(int(ri), int(ci))] = k
    assert len(cells) == 48, len(cells)
    for i, lab in enumerate(EV_ROWS):
        for j, ckey in enumerate(EV_COLS):
            k = cell[(lab, ckey)]
            assert cells[(i, j)] == ("X" if k is None else str(k)), (lab,
                                                                    ckey)

    # (2) slopes: gap == B/a seed rows in cr_figure_values.csv; tex == CSV.
    slope_rows = []
    for lab in TAX_LABELS:
        fam = STRONG[lab]
        for s in SEEDS:
            slope_rows.append((lab, s, pre[(lab, s)],
                               fam_test[(lab, fam, s)], fam))
    with open(os.path.join(DATADIR, "resplit_slopes.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20, len(rows)
    winnames = {"frequency_bucket": "Cadence", "metric_type": "MetricType",
                "domain": "Domain", "cardinality_bucket": "Cardinality"}
    texwins = {}
    with open(os.path.join(TIKZDIR, "resplit_slope_wins.tex")) as f:
        for line in f:
            if line.startswith("\\newcommand{\\slopeWins"):
                name = line.split("{\\slopeWins")[1].split("}")[0]
                texwins[name] = line.rsplit("{", 1)[1].rstrip("}\n")
    assert len(texwins) == 4, texwins
    ncoord = 0
    for lab in TAX_LABELS:
        texcoords = []
        with open(os.path.join(TIKZDIR,
                               "resplit_slope_%s.tex" % winnames[lab])) as f:
            for line in f:
                if ("coordinates" in line and "(0," in line
                        and "(1," in line):
                    a = line.split("(0,")[1].split(")")[0]
                    b = line.split("(1,")[1].split(")")[0]
                    texcoords.append((float(a), float(b)))
        # each resplit contributes one line + one dot pair -> dedupe
        uniq = [texcoords[i] for i in range(0, len(texcoords), 2)]
        assert len(uniq) == 5, (lab, len(uniq))
        fam = STRONG[lab]
        for s, (tt, rr) in zip(SEEDS, uniq):
            t, r = pre[(lab, s)], fam_test[(lab, fam, s)]
            ncoord += 1
            assert (abs(tt - round(t, 4)) < 1e-9
                    and abs(rr - round(r, 4)) < 1e-9), (lab, s)
    assert ncoord == 20, ncoord
    for (lab, s, t, r, fam) in slope_rows:
        key = ("B", "a", PRETTY[lab], "Toto minus %s" % FAM_SHORT[fam], s)
        assert key in cv, key
        assert abs(float(cv[key]) - (t - r)) < 5e-7, (key, cv[key], t - r)
        assert abs(float(_fmt6(t)) - t) < 1e-6
    # slope y-range assertion for the shared axis (0.3-1.0).
    for lab, s, t, r, fam in slope_rows:
        assert 0.3 <= t <= 1.0 and 0.3 <= r <= 1.0, (lab, s, t, r)
    # wins macros used in panel titles.
    for lab in TAX_LABELS:
        fam = STRONG[lab]
        w = wins([pre[(lab, s)] - fam_test[(lab, fam, s)] for s in SEEDS])
        assert texwins[winnames[lab]] == "%d/5" % w, (lab, texwins)

    # (3) interchange: dots/stats == B/b rows in cr_figure_values.csv.
    for model in ["Toto", "MOMENT"]:
        tag = model.lower()
        with open(os.path.join(DATADIR,
                               "interchange_%s_dots.csv" % tag)) as f:
            dots = list(csv.DictReader(f))
        assert len(dots) == 20, (model, len(dots))
        for d in dots:
            col = ("Probe check" if d["x"] == "Probe check"
                   else "beta " + d["x"].split()[-1])
            src, colname, filt = PANEL_B_SRC[(model, col)]
            key = ("B", "b", col, model, d["seed"])
            assert key in cv, key
            assert abs(float(d["value"]) - float(cv[key])) < 1e-9, key
        with open(os.path.join(DATADIR,
                               "interchange_%s_stats.csv" % tag)) as f:
            stats = list(csv.DictReader(f))
        assert len(stats) == 4, (model, len(stats))
        for st in stats:
            col = ("Probe check" if st["x"] == "Probe check"
                   else "beta " + st["x"].split()[-1])
            for stat in ["mean", "halfwidth"]:
                key = ("B", "b", col, model, stat)
                assert key in cv, key
                got = float(st[stat if stat == "mean" else "halfwidth"])
                assert abs(got - float(cv[key])) < 1e-9, (key, got,
                                                          cv[key])
    print("tikz data: 48 cells, 20 slopes, 40 dots, 8 stats "
          "all match cr_figure_values.csv")


if __name__ == "__main__":
    main()
