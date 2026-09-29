"""Reject malformed submissions before upload."""
import argparse, sys
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission", required=True)
    p.add_argument("--sample", required=True)
    a = p.parse_args()
    sub = pd.read_csv(a.submission)
    smp = pd.read_csv(a.sample)
    assert list(sub.columns) == ["clip_id", "label"], f"bad columns {list(sub.columns)}"
    assert len(sub) == len(smp), f"{len(sub)} rows, expected {len(smp)}"
    assert sub.clip_id.is_unique, "duplicate clip_id"
    assert set(sub.clip_id) == set(smp.clip_id), "clip_id mismatch vs sample"
    assert sub.label.between(0, 1).all(), "labels outside [0,1]"
    assert sub.label.notna().all(), "empty values"
    print("submission OK:", len(sub), "rows")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("INVALID:", e)
        sys.exit(1)
