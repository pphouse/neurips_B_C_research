#!/usr/bin/env python3
"""Build the BRCA1 variant table (parquet) from SGE data + UniProt protein."""
import argparse
import shutil
from pathlib import Path

from cdd.data.build_table import build
from cdd.utils.common import load_yaml, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    # ensure protein fasta present (fetch if missing)
    prot = Path(cfg["protein_fasta"])
    prot.parent.mkdir(parents=True, exist_ok=True)
    if not prot.exists():
        import urllib.request
        url = f"https://rest.uniprot.org/uniprotkb/{cfg['uniprot_id']}.fasta"
        urllib.request.urlretrieve(url, prot)

    df = build(cfg["sge_xlsx"], prot, seed=cfg.get("seed", 0))
    protein = df.attrs["protein"]

    out = Path(cfg["out_table"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    # summary
    summary = {
        "n_total": len(df),
        "n_missense_paired": int(df["paired"].sum()),
        "consequence_counts": df["consequence"].value_counts().to_dict(),
        "domain_counts": df["domain"].value_counts().to_dict(),
        "clinvar_bin_counts": df["clinvar_bin"].value_counts(dropna=False).to_dict(),
        "func_class_counts": df["func_class"].value_counts().to_dict(),
        "missense_by_domain": df[df.paired]["domain"].value_counts().to_dict(),
        "protein_len": len(protein),
    }
    save_json(summary, out.parent / "brca1_variants_summary.json")
    print(f"Wrote {out} ({len(df)} variants, {summary['n_missense_paired']} paired missense)")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
