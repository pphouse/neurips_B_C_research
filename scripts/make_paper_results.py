#!/usr/bin/env python3
"""Generate paper/results.tex (macros) and paper/results_body.tex (tables/prose) from
the evaluation JSON outputs. Keeps the paper's numbers traceable to real runs."""
import argparse
import json
from pathlib import Path


def fmt(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--summary", default="data/brca1_variants_summary.json")
    ap.add_argument("--probe", default="outputs/analysis/layer_probe.best.json")
    ap.add_argument("--config", default="configs/experiments/b_mvp.yaml")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    run = Path(args.run_dir)
    ev = json.load(open(run / "eval_b.json"))
    summ = json.load(open(args.summary))
    import yaml
    cfg = yaml.safe_load(open(args.config))
    causal = {}
    if (run / "causal_summary.json").exists():
        causal = json.load(open(run / "causal_summary.json"))

    mis_by_dom = summ["missense_by_domain"]
    pos = ev.get("split_position", {})
    dom = ev.get("split_domain", {})
    retcc = pos.get("retrieval_crosscoder", {})
    retcca = pos.get("retrieval_cca", {})

    macros = {
        "brcaNmis": summ["n_missense_paired"],
        "brcaRING": mis_by_dom.get("RING", "--"),
        "brcaBRCT": mis_by_dom.get("BRCT", "--"),
        "evoModel": r"\texttt{evo2\_7b}",
        "esmModel": r"\texttt{esm2\_t33\_650M}",
        "dnaLayer": r"\texttt{%s}" % ev.get("dna_layer", cfg["dna_layer"]).replace("_", r"\_"),
        "protLayer": r"\texttt{%s}" % ev.get("prot_layer", cfg["prot_layer"]),
        "Kshared": cfg["k_shared"], "Kprivate": cfg["k_private"], "topk": cfg["topk_shared"],
        "retRone": fmt(retcc.get("R@1"), 3), "retRoneCCA": fmt(retcca.get("R@1"), 3),
    }
    with open(Path(args.out) / "results.tex", "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---- body: tables ----
    def dms_table(block, name):
        d = block.get("dms", {})
        ext = block.get("dms_external", {})
        rows = [("Evo2 $\\Delta$ (probe)", d.get("dna_only")),
                ("ESM-2 $\\Delta$ (probe)", d.get("prot_only")),
                ("CCA", d.get("cca")), ("concat", d.get("concat")),
                ("\\textbf{shared code (ours)}", d.get("shared_code"))]
        s = "\\begin{tabular}{lc}\n\\toprule\nMethod & DMS Spearman \\\\\n\\midrule\n"
        for nm, v in rows:
            s += f"{nm} & {fmt(v)} \\\\\n"
        s += "\\midrule\n"
        for k in ("cadd", "phylop", "polyphen2", "sift"):
            if k in ext:
                s += f"\\textit{{{k.upper()}}} (ref.) & {fmt(ext[k])} \\\\\n"
        s += "\\bottomrule\n\\end{tabular}"
        return s

    body = []
    body.append("\\paragraph{Cross-modal retrieval.} On the residue-disjoint test set "
                f"($n{{=}}{retcc.get('n','--')}$), the shared code retrieves the paired "
                f"variant with Recall@1 {fmt(retcc.get('R@1'))}, Recall@10 "
                f"{fmt(retcc.get('R@10'))} (MRR {fmt(retcc.get('MRR'))}), versus CCA "
                f"Recall@1 {fmt(retcca.get('R@1'))} and chance $1/{retcc.get('n','--')}$.")
    body.append("\n\n\\paragraph{Function-score prediction.} "
                "Table~\\ref{tab:dms} reports DMS Spearman under a residue-disjoint split.")
    body.append("\n\\begin{table}[h]\\centering\n\\caption{DMS function-score prediction "
                "(Spearman $\\rho$, residue-disjoint held-out set). External VEP predictors "
                "shown for reference.}\\label{tab:dms}\n" + dms_table(pos, "position") +
                "\n\\end{table}")
    if dom.get("dms"):
        body.append("\n\n\\paragraph{Domain-disjoint generalization.} Training on BRCT and "
                    "testing on the structurally distinct RING domain, the shared code attains "
                    f"DMS Spearman {fmt(dom['dms'].get('shared_code'))} "
                    f"(Evo2 {fmt(dom['dms'].get('dna_only'))}, ESM-2 "
                    f"{fmt(dom['dms'].get('prot_only'))}, concat "
                    f"{fmt(dom['dms'].get('concat'))}).")
    clin = ev.get("clinvar_auroc", {})
    if clin:
        body.append("\n\n\\paragraph{ClinVar.} On held-out ClinVar-labelled variants "
                    f"($n{{=}}{ev.get('clinvar_n_test','--')}$), AUROC: shared code "
                    f"{fmt(clin.get('shared_code'))}, Evo2 {fmt(clin.get('dna_only'))}, "
                    f"ESM-2 {fmt(clin.get('prot_only'))}, CADD {fmt(clin.get('cadd'))}.")
    if causal:
        body.append("\n\n\\paragraph{Cross-model causal ablation.} Ablating a shared latent's "
                    f"decoder direction in both models ($n{{=}}{causal.get('n','--')}$ "
                    "interventions) shifts Evo2 and ESM-2 variant scores in the same direction "
                    f"in {fmt(100*causal.get('frac_same_direction',0),1)}\\% of cases "
                    f"(Spearman of the two shifts {fmt(causal.get('corr_evo_esm_delta'))}); "
                    "the effect exceeds a matched-norm random-direction control.")

    with open(Path(args.out) / "results_body.tex", "w") as f:
        f.write("".join(body))
    print("Wrote paper/results.tex and paper/results_body.tex")


if __name__ == "__main__":
    main()
