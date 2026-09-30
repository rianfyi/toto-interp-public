"""Data for the camera-ready TikZ Figures 1, 3 and 4.

Figures 1 and 3 are computed from the audited five-resplit control CSVs
(results/reviewer3_controls/ in the release, read through the staging path R3
described in the README); every Figure 1 mean and Student-t(4) 95% half-width
is asserted against Table 1 of the paper. Figure 4 is drawn from the
interchange tables that plot_camera_ready_figures.py writes to
paper/neurips2026/figures/data/. Outputs are small CSV tables in
paper/neurips2026/figures/data/ and TikZ snippets in
paper/neurips2026/figures/tikz/gen/.
"""
import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "runs/rebuttal/reviewer3_controls"
OUT = ROOT / "paper/neurips2026/figures/data"
T4 = 2.7764451051977987  # Student-t(4) 0.975 quantile

LABELS = [("frequency_bucket", "Cadence"), ("metric_type", "Metric type"),
          ("domain", "Domain"), ("cardinality_bucket", "Cardinality")]
FAMILIES = {"cnn": "CNN", "fno": "FNO", "transformer": "Patch Transformer", "gbdt": "GBDT"}
# Table 1 (mean, half-width) as printed in the paper, used only for assertions.
TABLE1 = {
    "Cadence": {"Toto": (0.766, 0.024), "Six-stat.": (0.409, 0.047), "Shuffled": (0.486, 0.011),
                "Random init.": (0.547, 0.040), "Block perm.": (0.567, 0.020), "raw": (0.633, 0.047)},
    "Metric type": {"Toto": (0.545, 0.037), "Six-stat.": (0.347, 0.030), "Shuffled": (0.321, 0.032),
                    "Random init.": (0.398, 0.035), "Block perm.": (0.420, 0.016), "raw": (0.498, 0.007)},
    "Domain": {"Toto": (0.484, 0.011), "Six-stat.": (0.371, 0.037), "Shuffled": (0.317, 0.037),
               "Random init.": (0.362, 0.017), "Block perm.": (0.394, 0.011), "raw": (0.482, 0.040)},
    "Cardinality": {"Toto": (0.471, 0.029), "Six-stat.": (0.635, 0.027), "Shuffled": (0.217, 0.029),
                    "Random init.": (0.250, 0.025), "Block perm.": (0.428, 0.127), "raw": (0.961, 0.016)},
}


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def mean_hw(values):
    values = [float(v) for v in values]
    assert len(values) == 5, values
    return statistics.mean(values), T4 * statistics.stdev(values) / math.sqrt(5)


def per_seed(table, label, column, **filters):
    out = {}
    for r in table:
        if r["label"] != label or any(r.get(k) != v for k, v in filters.items()):
            continue
        out[int(r["seed"])] = float(r[column])
    assert sorted(out) == [42, 43, 44, 45, 46], (label, column, filters, sorted(out))
    return [out[s] for s in sorted(out)]


def main():
    uncond = rows(R3 / "unconditional_selected_all_seeds.csv")
    perm = rows(R3 / "layer_permuted_selected_all_seeds.csv")
    raw = rows(R3 / "raw_control_all_seeds.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    fig1, slopes, slope_means = [], [], []
    for key, pretty in LABELS:
        series = {
            "Toto": per_seed(uncond, key, "test_macro_f1", weight_source="pretrained"),
            "Six-stat.": per_seed(uncond, key, "baseline_test_macro_f1", weight_source="pretrained"),
            "Shuffled": per_seed(uncond, key, "shuffled_test_macro_f1", weight_source="pretrained"),
            "Random init.": per_seed(uncond, key, "test_macro_f1", weight_source="random_init"),
            "Block perm.": per_seed(perm, key, "test_macro_f1"),
        }
        fam_val = {f: statistics.mean(per_seed(raw, key, "val_macro_f1", method=f)) for f in FAMILIES}
        best = max(fam_val, key=fam_val.get)  # strongest family by mean validation macro-F1
        series["raw"] = per_seed(raw, key, "test_macro_f1", method=best)
        for name, vals in series.items():
            m, h = mean_hw(vals)
            tm, th = TABLE1[pretty][name]
            assert round(m, 3) == tm and round(h, 3) == th, (pretty, name, m, h, tm, th)
            shown = f"Best raw window ({FAMILIES[best]})" if name == "raw" else name
            fig1.append({"label": pretty, "control": name, "shown": shown,
                         "mean": f"{m:.6f}", "halfwidth": f"{h:.6f}"})
        toto, rawv = series["Toto"], series["raw"]
        for seed, (t, r) in zip([42, 43, 44, 45, 46], zip(toto, rawv)):
            slopes.append({"label": pretty, "seed": seed, "toto": f"{t:.6f}", "raw": f"{r:.6f}",
                           "toto_higher": int(t > r), "family": FAMILIES[best]})
        slope_means.append({"label": pretty, "toto": f"{statistics.mean(toto):.6f}",
                            "raw": f"{statistics.mean(rawv):.6f}", "family": FAMILIES[best],
                            "wins": sum(t > r for t, r in zip(toto, rawv))})

    def write(name, table):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(table)

    write("fig1_controls.csv", fig1)
    write("fig3_slopes.csv", slopes)
    write("fig3_means.csv", slope_means)
    write_tikz(fig1, slopes, slope_means)
    print("ok: data CSVs and TikZ snippets written (all Table 1 values asserted)")


TIKZ = ROOT / "paper/neurips2026/figures/tikz/gen"
F1_ROWS = ["Toto", "raw", "Block perm.", "Random init.", "Six-stat.", "Shuffled"]  # top to bottom


def write_tikz(fig1, slopes, slope_means):
    """Emit pgfplots coordinate snippets (generated -- do not edit by hand)."""
    TIKZ.mkdir(parents=True, exist_ok=True)
    hdr = "% Generated by scripts/make_camera_ready_fig_data.py -- do not edit.\n"
    for k, (_, pretty) in enumerate(LABELS, start=1):
        rows_k = {r["control"]: r for r in fig1 if r["label"] == pretty}
        lines = [hdr]
        for name in F1_ROWS:
            r = rows_k[name]
            y = len(F1_ROWS) - F1_ROWS.index(name)
            style = {"Toto": "fig1 toto", "raw": "fig1 raw"}.get(name, "fig1 control")
            lines.append(f"\\addplot[{style}] coordinates {{({r['mean']},{y}) +- ({r['halfwidth']},0)}};\n")
        fam = rows_k["raw"]["shown"].split("(")[1].rstrip(")")
        r = rows_k["raw"]
        m, h = float(r["mean"]), float(r["halfwidth"])
        # family name to the left of its marker (clear of Toto's reference line)
        lines.append(f"\\node[fig1 fam, anchor=east] at (axis cs:{m - h - 0.015:.4f},5) {{{fam}}};\n")
        t = float(rows_k["Toto"]["mean"])
        lines.insert(1, f"\\draw[fig1 ref] (axis cs:{t:.6f},0.4) -- (axis cs:{t:.6f},6.6);\n")
        (TIKZ / f"fig1_panel{k}.tex").write_text("".join(lines))
    for k, (_, pretty) in enumerate(LABELS, start=1):
        lines = [hdr]
        for r in (x for x in slopes if x["label"] == pretty):
            style = "fig3 up" if r["toto_higher"] else "fig3 down"
            lines.append(f"\\addplot[{style}] coordinates {{(0,{r['toto']}) (1,{r['raw']})}};\n")
        m = next(x for x in slope_means if x["label"] == pretty)
        lines.append(f"\\addplot[fig3 mean] coordinates {{(0,{m['toto']}) (1,{m['raw']})}};\n")
        (TIKZ / f"fig3_panel{k}.tex").write_text("".join(lines))
    letters = "ABCD"
    defs = [hdr]
    for k, m in enumerate(slope_means):
        defs.append(f"\\def\\figThreeFam{letters[k]}{{{m['family']}}}\\def\\figThreeWins{letters[k]}{{{m['wins']}}}\n")
    (TIKZ / "fig3_defs.tex").write_text("".join(defs))
    # Figure 4: probe check (constant across blends by construction) and forecast endpoint.
    beta = {"2": 0.25, "3": 0.5, "4": 1.0}
    for model, tag in [("toto", "toto"), ("moment", "moment")]:
        stats = rows(ROOT / f"paper/neurips2026/figures/data/interchange_{model}_stats.csv")
        dots = rows(ROOT / f"paper/neurips2026/figures/data/interchange_{model}_dots.csv")
        probe = next(r for r in stats if r["xpos"] == "1")
        pm, ph = float(probe["mean"]), float(probe["halfwidth"])
        lines = [hdr,
                 f"\\fill[fig4 band] (axis cs:0.08,{pm - ph:.6f}) rectangle (axis cs:1.14,{pm + ph:.6f});\n",
                 f"\\draw[fig4 probe] (axis cs:0.08,{pm:.6f}) -- (axis cs:1.14,{pm:.6f});\n",
                 f"\\node[fig4 lab, anchor=south east] at (axis cs:1.14,{pm + ph:.6f}) {{probe check (same at every $\\beta$)}};\n"]
        pts = [f"({beta[d['xpos']] + (int(d['seed']) - 44) * 0.018:.4f},{float(d['value']):.6f})"
               for d in dots if d["xpos"] in beta]
        lines.append(f"\\addplot[fig4 dots] coordinates {{{' '.join(pts)}}};\n")
        means = [f"({beta[r['xpos']]},{float(r['mean']):.6f}) +- (0,{float(r['halfwidth']):.6f})"
                 for r in stats if r["xpos"] in beta]
        lines.append(f"\\addplot[fig4 mean] coordinates {{{' '.join(means)}}};\n")
        (TIKZ / f"fig4_{tag}.tex").write_text("".join(lines))


if __name__ == "__main__":
    main()
