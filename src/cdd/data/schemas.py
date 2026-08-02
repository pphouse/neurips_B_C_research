"""Data schemas for variants and activation records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class VariantRecord:
    """A single-nucleotide variant with genomic + protein coordinates and labels.

    All coordinates are 1-based. Genomic coords are GRCh37/hg19 (matching the
    Findlay 2018 BRCA1 SGE dataset). Protein coords are on the MANE/canonical
    transcript (BRCA1 NM_007294.4 / UniProt P38398, 1863 aa).
    """

    variant_id: str
    gene: str
    chrom: str
    pos: int           # genomic position (hg19, 1-based)
    ref: str           # genomic reference allele (+ strand)
    alt: str           # genomic alt allele (+ strand)
    # functional / clinical labels
    dms_score: Optional[float] = None      # SGE function.score.mean (higher = more functional)
    func_class: Optional[str] = None       # FUNC / INT / LOF
    clinvar: Optional[str] = None          # pathogenic / benign / vus / None
    # consequence / protein mapping (from VEP)
    consequence: Optional[str] = None      # missense_variant, synonymous_variant, splice_*, ...
    aa_ref: Optional[str] = None           # WT amino acid (1-letter)
    aa_alt: Optional[str] = None           # mutant amino acid (1-letter)
    aa_pos: Optional[int] = None           # 1-based residue position on protein
    strand: Optional[int] = None           # gene strand: +1 / -1
    # mechanism annotations (filled later)
    domain: Optional[str] = None           # RING / BRCT / linker / None
    splice_dist: Optional[int] = None      # distance (bp) to nearest exon boundary
    conservation: Optional[float] = None
    reason_code: Optional[str] = None       # why a variant was dropped, if any

    def to_dict(self) -> dict:
        return asdict(self)


# Category flags derived from consequence
MISSENSE = "missense_variant"
SYNONYMOUS = "synonymous_variant"
SPLICE = ("splice_donor_variant", "splice_acceptor_variant", "splice_region_variant")
STOP_GAINED = "stop_gained"


def is_missense(c: Optional[str]) -> bool:
    return c is not None and "missense" in c


def is_splice(c: Optional[str]) -> bool:
    return c is not None and "splice" in c
