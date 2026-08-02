"""Data integrity + no-leakage tests (CPU, no GPU)."""
import numpy as np
import pandas as pd
import pytest

TABLE = "data/brca1_variants.parquet"


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(TABLE)


def test_paired_are_missense(df):
    assert (df["paired"] == df["is_missense"]).all()
    assert df["paired"].sum() > 1500


def test_aa_mapping_present(df):
    mis = df[df.paired]
    assert mis["aa_pos"].notna().all()
    assert mis["aa_ref"].notna().all()
    assert (mis["aa_ref"].str.len() == 1).all()


def test_position_split_disjoint(df):
    """No residue (aa_pos) appears in both train and test for the paired missense set."""
    mis = df[df.paired]
    g = mis.groupby("aa_pos")["split_position"].nunique()
    assert (g == 1).all(), "residue leakage across split_position in missense set"


def test_domain_split_disjoint(df):
    """Domain split: RING is test, BRCT is train, never overlap."""
    tr = set(df[df.split_domain == "train"].domain.unique())
    te = set(df[df.split_domain == "test"].domain.unique())
    assert tr == {"BRCT"} and te == {"RING"}


def test_variant_ids_unique(df):
    assert df["variant_id"].is_unique
