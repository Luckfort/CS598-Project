"""
python download_data.py

Should create:

    raw/2wiki_dev.json                2WikiMultihopQA, dev split
    raw/musique_full_v1.0_dev.jsonl   MuSiQue-Full, dev split
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

MUSIQUE = ("bdsaglam/musique", "musique_full_v1.0_dev.jsonl")
TWOWIKI = ("xanhho/2WikiMultihopQA", "dev.parquet")
NESTED = ("context", "supporting_facts", "evidences")

def _unpack(value):
    """Accept either shape: the mirror's JSON strings, or real nested lists if
    you point this at the original release instead."""
    if isinstance(value, str):
        return json.loads(value)
    if hasattr(value, "tolist"):          # numpy array from parquet
        return value.tolist()
    return value


def fetch_musique(out_dir: Path, force: bool) -> Path:
    
    dest = out_dir / "musique_full_v1.0_dev.jsonl"
    if dest.exists() and not force:
        print(f"  {dest} already there, skipping")
        return dest
    src = hf_hub_download(MUSIQUE[0], MUSIQUE[1], repo_type="dataset")
    dest.write_bytes(Path(src).read_bytes())
    n = sum(1 for line in dest.open(encoding="utf-8") if line.strip())
    print(f"  {dest}  ({n:,} questions)")
    return dest


def fetch_2wiki(out_dir: Path, force: bool) -> Path:

    dest = out_dir / "2wiki_dev.json"
    if dest.exists() and not force:
        print(f"  {dest} already there, skipping")
        return dest
    src = hf_hub_download(TWOWIKI[0], TWOWIKI[1], repo_type="dataset")
    rows = []
    for _, r in pd.read_parquet(src).iterrows():
        row = {"_id": r["_id"], "type": r["type"], "question": r["question"],
               "answer": r["answer"]}
        row.update({c: _unpack(r[c]) for c in NESTED})
        rows.append(row)
    dest.write_text(json.dumps(rows), encoding="utf-8")
    print(f"  {dest}  ({len(rows):,} questions)")
    return dest

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("raw"))
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file is already there")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading into {args.out_dir}/ (about 115 MB, a few minutes)")
    try:
        fetch_musique(args.out_dir, args.force)
        fetch_2wiki(args.out_dir, args.force)
    except Exception as e:
        sys.exit(f"\ndownload failed: {e}\n"
                 f"If this is a network or HuggingFace outage, try again. The "
                 f"files can also be fetched by hand from the dataset pages: "
                 f"{MUSIQUE[0]} and {TWOWIKI[0]}.")

    print("\ndone. next:\n"
          "  python prepare_data.py \\\n"
          f"    --dataset 2wiki    {args.out_dir}/2wiki_dev.json \\\n"
          f"    --dataset musique  {args.out_dir}/musique_full_v1.0_dev.jsonl \\\n"
          "    --limit 500 --out data")

if __name__ == "__main__":
    main()
