"""Recompute the paper's reported results (tables, plotted figure values,
in-text statistics) from the packaged results CSVs.

Reads ONLY results/ (audited five-resplit artifacts) and checks each value
against reproduce/expected_numbers.csv.

Conventions (same as the paper, Sec. Method):
  mean         = arithmetic mean over the five resplit-level values
  half-width   = Student-t(4) two-sided 95% half-width across the five values
  win          = a resplit with a strictly positive paired difference
  strongest    = raw-window family with the highest MEAN VALIDATION macro-F1

Usage:  python reproduce/verify_paper_numbers.py
Exit code 0 iff every check passes.
"""

import csv
import json
import math
import os
import statistics
import sys

from scipy.stats import t as t_dist

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RES = os.path.join(PKG, "results")
R3 = os.path.join(RES, "reviewer3_controls")
R5 = os.path.join(RES, "reviewer_replications_5seed")

SEEDS = ["42", "43", "44", "45", "46"]
T95 = float(t_dist.ppf(0.975, 4))

LABELS = ["frequency_bucket", "metric_type", "domain", "cardinality_bucket"]
SHORT = {"frequency_bucket": "cad", "metric_type": "met",
         "domain": "dom", "cardinality_bucket": "card"}
FAMS = ["cnn", "fno", "transformer", "gbdt_spectral_autocorrelation_last_patch"]
BLENDS = ["0.25", "0.5", "1.0"]
BSHORT = {"0.25": "025", "0.5": "05", "1.0": "10"}
COND2LABEL = {"frequency_given_infra_gauge": "frequency_bucket",
              "metric_type_given_app_short": "metric_type",
              "domain_given_gauge_short": "domain"}
COND_SHORT = {"frequency_given_infra_gauge": "cad",
              "metric_type_given_app_short": "met",
              "domain_given_gauge_short": "dom"}


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_hw(xs):
    xs = [float(x) for x in xs]
    assert len(xs) == 5, "expected five resplit values, got %d" % len(xs)
    m = statistics.mean(xs)
    hw = T95 * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, hw


def wins(gaps):
    return sum(float(g) > 0 for g in gaps)


C = {}  # computed key -> value


def put(key, value):
    assert key not in C, "duplicate key " + key
    C[key] = value


# ------------------------------------------------------- taxonomy-control inputs
un = load(os.path.join(R3, "unconditional_selected_all_seeds.csv"))
pre, rnd, rawbase, shufbase = {}, {}, {}, {}
pre_acc = {}
for r in un:
    k = (r["label"], r["seed"])
    if r["weight_source"] == "pretrained":
        pre[k] = float(r["test_macro_f1"])
        rawbase[k] = float(r["baseline_test_macro_f1"])
        shufbase[k] = float(r["shuffled_test_macro_f1"])
        pre_acc[k] = float(r["test_accuracy"])
    elif r["weight_source"] == "random_init":
        rnd[k] = float(r["test_macro_f1"])

rc = load(os.path.join(R3, "raw_control_all_seeds.csv"))
fam_val, fam_test, fam_acc = {}, {}, {}
for r in rc:
    fam_val.setdefault((r["label"], r["raw_window_family"]), []).append(
        float(r["val_macro_f1"]))
    fam_test.setdefault((r["label"], r["raw_window_family"]), []).append(
        float(r["test_macro_f1"]))
    fam_acc.setdefault((r["label"], r["raw_window_family"]), []).append(
        float(r["test_accuracy"]))
STRONG = {}
for lab in LABELS:
    fams = sorted({f for (l, f) in fam_val if l == lab})
    STRONG[lab] = max(fams, key=lambda f: statistics.mean(fam_val[(lab, f)]))

lp = load(os.path.join(R3, "layer_permuted_selected_all_seeds.csv"))
permd = {(r["label"], r["seed"]): float(r["test_macro_f1"]) for r in lp}

co = load(os.path.join(R3, "conditional_all_seeds.csv"))
cond_toto, cond_ri, cond_raw, cond_sup = {}, {}, {}, {}
for r in co:
    key = (r["conditional_key"], r["seed"])
    if r["weight_source"] == "pretrained":
        cond_toto[key] = float(r["test_macro_f1"])
        cond_raw[key] = float(r["baseline_test_macro_f1"])
        cond_sup[key] = json.loads(r["balanced_series_counts"])["test"]
    else:
        cond_ri[key] = float(r["test_macro_f1"])

ho = load(os.path.join(R3, "structural_holdout_all_seeds.csv"))

cv = load(os.path.join(R3, "pairwise_cramers_v_all_seeds.csv"))

with open(os.path.join(R3, "reviewer3_summary.json")) as f:
    summary = json.load(f)
param_mean = {(r["label"], r["method"]): float(r["parameter_count_mean"])
              for r in summary["raw_control_summary"]}

# ------------------------------------------------------- replication inputs
pp = {(r["split_seed"], r["blend"]): r
      for r in load(os.path.join(R5, "paired_patch_per_resplit.csv"))}
mp = {(r["label"], r["split_seed"]): r
      for r in load(os.path.join(R5, "moment_per_resplit.csv"))}
mr = {(r["label"], r["split_seed"]): float(r["random_test_score"])
      for r in load(os.path.join(
          R5, "moment_random_structural_per_resplit.csv"))}
md = {(r["label"], r["split_seed"]): r
      for r in load(os.path.join(R5, "moment_dynamic_per_resplit.csv"))}
mi = {(r["split_seed"], r["blend"]): r
      for r in load(os.path.join(R5, "moment_interchange_per_resplit.csv"))}
tr = load(os.path.join(R5, "transfer_per_resplit.csv"))
mom_man = load(os.path.join(R5, "moment_manifest.csv"))
pp_man = load(os.path.join(R5, "paired_patch_manifest.csv"))

# ------------------------------------------------------- Tables 1 and 10
FAM_PRETTY = {"fno": "FNO", "cnn": "CNN", "transformer": "PatchTransformer",
              "gbdt_spectral_autocorrelation_last_patch": "GBDT"}
for lab in LABELS:
    s = SHORT[lab]
    for col, d in (("toto", pre), ("raw", rawbase), ("shuf", shufbase),
                   ("ri", rnd)):
        vals = [d[(lab, sd)] for sd in SEEDS]
        m, hw = mean_hw(vals)
        put("T1A.%s.%s.mean" % (s, col), m)
        put("T1A.%s.%s.hw" % (s, col), hw)
    vals = [permd[(lab, sd)] for sd in SEEDS]
    m, hw = mean_hw(vals)
    put("T1B.%s.perm.mean" % s, m)
    put("T1B.%s.perm.hw" % s, hw)
    put("T1B.%s.fam" % s, FAM_PRETTY[STRONG[lab]])
    fam = STRONG[lab]
    # order rows by seed for determinism
    fvals = [float(next(r["test_macro_f1"] for r in rc
                        if r["label"] == lab and r["raw_window_family"] == fam
                        and r["seed"] == sd)) for sd in SEEDS]
    m, hw = mean_hw(fvals)
    put("T1B.%s.fammean" % s, m)
    put("T1B.%s.famhw" % s, hw)
    gaps = [pre[(lab, sd)] - fvals[i] for i, sd in enumerate(SEEDS)]
    m, hw = mean_hw(gaps)
    put("T1B.%s.gap.mean" % s, m)
    put("T1B.%s.gap.hw" % s, hw)
    put("T1B.%s.gap.wins" % s, wins(gaps))
    for fam2 in FAMS:
        fvals2 = [float(next(r["test_macro_f1"] for r in rc
                             if r["label"] == lab
                             and r["raw_window_family"] == fam2
                             and r["seed"] == sd)) for sd in SEEDS]
        m2, hw2 = mean_hw(fvals2)
        put("T10.%s.%s.mean" % (s, FAM_PRETTY[fam2]), m2)
        put("T10.%s.%s.hw" % (s, FAM_PRETTY[fam2]), hw2)
    for col, key in (("toto", "T1A"), ("ri", "T1A"), ("perm", "T1B")):
        if col == "toto":
            put("T10.%s.toto.mean" % s, C["T1A.%s.toto.mean" % s])
            put("T10.%s.toto.hw" % s, C["T1A.%s.toto.hw" % s])
        elif col == "ri":
            put("T10.%s.ri.mean" % s, C["T1A.%s.ri.mean" % s])
            put("T10.%s.ri.hw" % s, C["T1A.%s.ri.hw" % s])
        else:
            put("T10.%s.perm.mean" % s, C["T1B.%s.perm.mean" % s])
            put("T10.%s.perm.hw" % s, C["T1B.%s.perm.hw" % s])

# Domain accuracy sentence (Sec. 5.1)
acc_toto = [pre_acc[("domain", sd)] for sd in SEEDS]
acc_gbdt = [fam_acc[("domain",
                     "gbdt_spectral_autocorrelation_last_patch")][i]
            for i in range(5)]
m, hw = mean_hw(acc_toto)
put("ACC.dom.toto.mean", m)
m, hw = mean_hw(acc_gbdt)
put("ACC.dom.gbdt.mean", m)
put("ACC.dom.wins", wins([a - b for a, b in zip(acc_toto, acc_gbdt)]))

# ------------------------------------------------------- Table 2 conditional
for ckey, lab in COND2LABEL.items():
    s = COND_SHORT[ckey]
    tv = [cond_toto[(ckey, sd)] for sd in SEEDS]
    rv = [cond_ri[(ckey, sd)] for sd in SEEDS]
    bv = [cond_raw[(ckey, sd)] for sd in SEEDS]
    for col, vals in (("toto", tv), ("ri", rv), ("raw", bv)):
        m, hw = mean_hw(vals)
        put("T2.%s.%s.mean" % (s, col), m)
        put("T2.%s.%s.hw" % (s, col), hw)
    put("T2.%s.wins_ri" % s, wins([a - b for a, b in zip(tv, rv)]))
    put("T2.%s.wins_raw" % s, wins([a - b for a, b in zip(tv, bv)]))
    sup = [list(cond_sup[(ckey, sd)].values())[0] for sd in SEEDS]
    put("T2SUP.%s.min" % s, min(sup))
    put("T2SUP.%s.max" % s, max(sup))

# ------------------------------------------------------- Table 3 MOMENT taxonomy
for lab in LABELS:
    s = SHORT[lab]
    mv = [float(mp[(lab, sd)]["test_macro_f1"]) for sd in SEEDS]
    rv = [mr[(lab, sd)] for sd in SEEDS]
    bv = [float(mp[(lab, sd)]["raw_test_macro_f1"]) for sd in SEEDS]
    for col, vals in (("mom", mv), ("ri", rv), ("raw", bv)):
        m, hw = mean_hw(vals)
        put("T3.%s.%s.mean" % (s, col), m)
        put("T3.%s.%s.hw" % (s, col), hw)
    gaps = [a - b for a, b in zip(mv, bv)]
    m, hw = mean_hw(gaps)
    put("T3.%s.gap.mean" % s, m)
    put("T3.%s.gap.hw" % s, hw)
    put("T3.%s.gap.wins" % s, wins(gaps))
    gaps_ri = [a - b for a, b in zip(mv, rv)]
    m, hw = mean_hw(gaps_ri)
    put("T3GAPRI.%s.mean" % s, m)
    put("T3GAPRI.%s.hw" % s, hw)
    put("T3GAPRI.%s.wins" % s, wins(gaps_ri))

# ------------------------------------------------------- Table 6 Toto donor exchange
for b in BLENDS:
    s = BSHORT[b]
    bv = [float(pp[(sd, b)]["burst_win_fraction"]) for sd in SEEDS]
    wv = [float(pp[(sd, b)]["wape_real_over_null_median"]) for sd in SEEDS]
    m, hw = mean_hw(bv)
    put("T4.%s.burst.mean" % s, m)
    put("T4.%s.burst.hw" % s, hw)
    put("T4.%s.cnt" % s, sum(v > 0.5 for v in bv))
    m, hw = mean_hw(wv)
    put("T4.%s.wape.mean" % s, m)
    put("T4.%s.wape.hw" % s, hw)
pv = [float(pp[(sd, "0.25")]["probe_win_fraction"]) for sd in SEEDS]
m, hw = mean_hw(pv)
put("T4PROBE.mean", m)
put("T4PROBE.hw", hw)
put("T4PROBE.wins", sum(v > 0.5 for v in pv))
r2p = [float(pp[(sd, "0.25")]["probe_test_r2"]) for sd in SEEDS]
r2r = [float(pp[(sd, "0.25")]["probe_raw_test_r2"]) for sd in SEEDS]
m, hw = mean_hw(r2p)
put("T4R2.probe.mean", m)
put("T4R2.probe.hw", hw)
m, hw = mean_hw(r2r)
put("T4R2.raw.mean", m)
put("T4R2.raw.hw", hw)
put("T4R2.wins", wins([a - b for a, b in zip(r2p, r2r)]))

# ------------------------------------------------------- Table 7 external transfer
for bench in ("fev", "lsf"):
    vals = [float(r["transfer_r2_macro"]) for r in tr
            if r["label"] == "coordination" and r["benchmark"] == bench]
    assert len(vals) == 5, bench
    m, hw = mean_hw(vals)
    put("T5.%s.mean" % bench, m)
    put("T5.%s.hw" % bench, hw)
    put("T5.%s.n" % bench,
        int(next(r["dataset_count"] for r in tr
                 if r["label"] == "coordination" and r["benchmark"] == bench)))

# ------------------------------------------------------- Table 4 raw-window params (0.1k)
put("T6.cnn.k", param_mean[("frequency_bucket", "cnn")] / 1000.0)
put("T6.fno.k", param_mean[("frequency_bucket", "fno")] / 1000.0)
trx_methods = [k for k in param_mean if k[0] == "frequency_bucket"
               and "transformer" in k[1]]
assert len(trx_methods) == 1, trx_methods
put("T6.trx.k", param_mean[trx_methods[0]] / 1000.0)

# ------------------------------------------------------- Table 5 held-out support
HO_ROWS = {("frequency_bucket", '["domain", "metric_type"]'): "cad_comb",
           ("frequency_bucket", '["domain"]'): "cad_dom",
           ("metric_type", '["domain", "frequency_bucket"]'): "met_comb",
           ("metric_type", '["domain"]'): "met_dom"}
for (tgt, axes), s in HO_ROWS.items():
    for src, kk in (("pretrained", "toto"), ("random_init", "ri")):
        vals = [float(r["test_macro_f1"]) for r in ho
                if r["target"] == tgt and r["holdout_axes"] == axes
                and r["weight_source"] == src]
        assert len(vals) == 5, (tgt, axes, src)
        put("T7.%s.%s.mean" % (s, kk), statistics.mean(vals))
        put("T7.%s.%s.min" % (s, kk), min(vals))
        put("T7.%s.%s.max" % (s, kk), max(vals))

# ------------------------------------------------------- Table 8 MOMENT dynamic
DYN = {"coordination": "coord", "current_burstiness": "curb",
       "future_burstiness": "futb", "shift_risk": "shift"}
for lab, s in DYN.items():
    mv = [float(md[(lab, sd)]["test_r2"]) for sd in SEEDS]
    bv = [float(md[(lab, sd)]["raw_test_r2"]) for sd in SEEDS]
    m, hw = mean_hw(mv)
    put("T8.%s.mom.mean" % s, m)
    put("T8.%s.mom.hw" % s, hw)
    m, hw = mean_hw(bv)
    put("T8.%s.raw.mean" % s, m)
    put("T8.%s.raw.hw" % s, hw)
    put("T8.%s.wins" % s, wins([a - b for a, b in zip(mv, bv)]))
mrd = {(r["label"], r["split_seed"]): r for r in load(os.path.join(
    R5, "moment_random_dynamic_per_resplit.csv"))}
for lab, s in DYN.items():
    rv = [float(mrd[(lab, sd)]["random_test_score"]) for sd in SEEDS]
    m, hw = mean_hw(rv)
    put("T8.%s.ri.mean" % s, m)
    put("T8.%s.ri.hw" % s, hw)

# ------------------------------------------------------- Table 9 MOMENT interchange
for b in BLENDS:
    s = BSHORT[b]
    fv = [float(mi[(sd, b)]["future_mae_real_better_fraction"])
          for sd in SEEDS]
    m, hw = mean_hw(fv)
    put("T9.%s.mae.mean" % s, m)
    put("T9.%s.mae.hw" % s, hw)
    put("T9.%s.cnt" % s, sum(v > 0.5 for v in fv))
mpv = [float(mi[(sd, "0.25")]["probe_win_fraction"]) for sd in SEEDS]
m, hw = mean_hw(mpv)
put("T9PROBE.mean", m)
put("T9PROBE.hw", hw)
put("T9PROBE.wins", sum(v > 0.5 for v in mpv))

# ------------------------------------------------------- Cramer's V trio (Sec. 5.2)
CV_PAIRS = {(("domain", "metric_type"), ("metric_type", "domain")): "dom_met",
            (("domain", "frequency_bucket"),
             ("frequency_bucket", "domain")): "dom_cad",
            (("metric_type", "frequency_bucket"),
             ("frequency_bucket", "metric_type")): "met_cad"}
for pair, s in CV_PAIRS.items():
    vals = [float(r["cramers_v"]) for r in cv
            if (r["left_label"], r["right_label"]) in pair
            and r["split"] == "all"]
    assert len(vals) == 5, (s, len(vals))
    m, hw = mean_hw(vals)
    put("CRAMER.%s.mean" % s, m)
    put("CRAMER.%s.hw" % s, hw)

# ------------------------------------------------------- derived in-text margins
put("MARGIN.cad", C["T2.cad.toto.mean"] - C["T2.cad.raw.mean"])
put("MARGIN.met", C["T2.met.toto.mean"] - C["T2.met.raw.mean"])
put("MARGIN.dom", C["T2.dom.toto.mean"] - C["T2.dom.raw.mean"])
put("GAP100.cad", C["T1B.cad.gap.mean"] * 100.0)
put("GAP100.met", C["T1B.met.gap.mean"] * 100.0)

# ------------------------------------------------------- win-count summary (supplementary)
FIG1 = {}
for lab in LABELS:
    s = SHORT[lab]
    FIG1[(s, "raw")] = wins([pre[(lab, sd)] - rawbase[(lab, sd)]
                             for sd in SEEDS])
    fam = STRONG[lab]
    FIG1[(s, "strongest")] = wins(
        [pre[(lab, sd)] - float(next(
            r["test_macro_f1"] for r in rc if r["label"] == lab
            and r["raw_window_family"] == fam and r["seed"] == sd))
         for sd in SEEDS])
    FIG1[(s, "randinit")] = wins([pre[(lab, sd)] - rnd[(lab, sd)]
                                  for sd in SEEDS])
    FIG1[(s, "blockperm")] = wins([pre[(lab, sd)] - permd[(lab, sd)]
                                   for sd in SEEDS])
    gri = [float(mp[(lab, sd)]["test_macro_f1"]) - mr[(lab, sd)]
           for sd in SEEDS]
    graw = [float(mp[(lab, sd)]["test_macro_f1"])
            - float(mp[(lab, sd)]["raw_test_macro_f1"]) for sd in SEEDS]
    FIG1[(s, "moment")] = min(wins(gri), wins(graw))
for ckey, lab in COND2LABEL.items():
    s = COND_SHORT[ckey]
    gri = [cond_toto[(ckey, sd)] - cond_ri[(ckey, sd)] for sd in SEEDS]
    graw = [cond_toto[(ckey, sd)] - cond_raw[(ckey, sd)] for sd in SEEDS]
    FIG1[(s, "strat")] = min(wins(gri), wins(graw))
FIG1[("card", "strat")] = "--"
FIG1[("futb", "raw")] = wins(
    [float(pp[(sd, "0.25")]["probe_test_r2"])
     - float(pp[(sd, "0.25")]["probe_raw_test_r2"]) for sd in SEEDS])
FIG1[("coord", "raw")] = "--"
for s in ("futb", "coord"):
    FIG1[(s, "strongest")] = "--"
    FIG1[(s, "randinit")] = "--"
    FIG1[(s, "blockperm")] = "--"
    FIG1[(s, "strat")] = "--"
for lab in ("coord", "futb"):
    key = "coordination" if lab == "coord" else "future_burstiness"
    FIG1[(lab, "moment")] = wins(
        [float(md[(key, sd)]["test_r2"])
         - float(md[(key, sd)]["raw_test_r2"]) for sd in SEEDS])
donor_counts = [sum(float(pp[(sd, b)]["burst_win_fraction"]) > 0.5
                    for sd in SEEDS) for b in BLENDS]
FIG1[("futb", "donor")] = max(donor_counts)
for s in ("cad", "met", "dom", "card", "coord"):
    FIG1[(s, "donor")] = "--"
tr_counts = []
for bench in ("fev", "lsf"):
    vals = [float(r["transfer_r2_macro"]) for r in tr
            if r["label"] == "coordination" and r["benchmark"] == bench]
    tr_counts.append(sum(v > 0 for v in vals))
FIG1[("coord", "transfer")] = max(tr_counts)
for s in ("cad", "met", "dom", "card", "futb"):
    FIG1[(s, "transfer")] = "--"
for key, val in FIG1.items():
    put("FIG1.%s.%s" % key, val)

# ------------------------------------------------------- Figures 3 and 4 values
for lab in LABELS:
    s = SHORT[lab]
    fam = STRONG[lab]
    for i, sd in enumerate(SEEDS):
        fval = float(next(r["test_macro_f1"] for r in rc
                          if r["label"] == lab
                          and r["raw_window_family"] == fam
                          and r["seed"] == sd))
        put("FIG2A.%s.seed%s" % (s, sd), pre[(lab, sd)] - fval)
for i, sd in enumerate(SEEDS):
    put("FIG2B.toto.probe.seed%s" % sd,
        float(pp[(sd, "0.25")]["probe_win_fraction"]))
    put("FIG2B.mom.probe.seed%s" % sd,
        float(mi[(sd, "0.25")]["probe_win_fraction"]))
    for b in BLENDS:
        put("FIG2B.toto.b%s.seed%s" % (BSHORT[b], sd),
            float(pp[(sd, b)]["burst_win_fraction"]))
        put("FIG2B.mom.b%s.seed%s" % (BSHORT[b], sd),
            float(mi[(sd, b)]["future_mae_real_better_fraction"]))

# ------------------------------------------------------- appendix series/window counts
for col, kk in (("train_series_count", "train"),
                ("val_series_count", "val"),
                ("test_series_count", "test")):
    vals = [int(r[col]) for r in mom_man]
    assert len(vals) == 5
    put("APP.series.%s" % kk, vals[0] if len(set(vals)) == 1 else -1)
put("APP.win.train.min", min(int(r["train_window_count"]) for r in mom_man))
put("APP.win.train.max", max(int(r["train_window_count"]) for r in mom_man))
put("APP.win.val.min", min(int(r["val_window_count"]) for r in mom_man))
put("APP.win.val.max", max(int(r["val_window_count"]) for r in mom_man))
put("APP.win.test.min", min(int(r["test_window_count"]) for r in mom_man))
put("APP.win.test.max", max(int(r["test_window_count"]) for r in mom_man))
pool = [int(r["n_windows"]) for r in pp_man]
put("APP.win.pool.min", min(pool))
put("APP.win.pool.max", max(pool))

# ------------------------------------------------------- check runner
TOL = {"m3": 5.1e-4, "h3": 5.1e-4, "f6": 1.1e-6, "k1": 5.1e-2,
       "about2": 5.1e-3, "int": 0.0, "str": 0.0}


def main():
    exp_path = os.path.join(HERE, "expected_numbers.csv")
    rows = load(exp_path)
    npass, nfail = 0, 0
    failures = []
    for r in rows:
        key, loc = r["key"], r["location"]
        kind, exp_raw = r["kind"], r["expected"]
        if key not in C:
            nfail += 1
            failures.append((key, loc, "MISSING from computed dict", exp_raw))
            print("FAIL %-28s %-22s missing computation" % (key, loc))
            continue
        got = C[key]
        if kind in ("int",):
            ok = int(got) == int(exp_raw)
            detail = "got %s expected %s" % (got, exp_raw)
        elif kind == "str":
            ok = str(got) == exp_raw
            detail = "got %r expected %r" % (got, exp_raw)
        else:
            tol = TOL[kind]
            ok = abs(float(got) - float(exp_raw)) <= tol
            detail = "got %.6f expected %s tol %g" % (got, exp_raw, tol)
        if ok:
            npass += 1
        else:
            nfail += 1
            failures.append((key, loc, detail, exp_raw))
            print("FAIL %-28s %-22s %s  [%s]" % (key, loc, detail,
                                                 r.get("description", "")))
    print("=" * 70)
    print("PASS %d  FAIL %d  TOTAL %d" % (npass, nfail, npass + nfail))
    if nfail:
        print("FAILURES:")
        for key, loc, detail, _ in failures:
            print("  %s (%s): %s" % (key, loc, detail))
        return 1
    print("ALL PAPER NUMBERS VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
