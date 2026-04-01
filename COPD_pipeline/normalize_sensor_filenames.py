from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")


def _read_csv(path: Path):
    for enc in ["cp949", "utf-8", "utf-8-sig"]:
        try:
            return pd.read_csv(path, encoding=enc, nrows=1)
        except Exception:
            continue
    return None


def infer_type(path: Path):
    stem = path.stem.upper()
    if "1X1" in stem:
        return "1X1"
    if "4X4" in stem:
        return "4X4"
    if "GAS" in stem:
        return "GAS"

    df = _read_csv(path)
    if df is None:
        return None

    cols = [c.upper() for c in df.columns]
    if "TIME" in cols and "PROCESS" in cols:
        if any(col.startswith("1_R") for col in cols) and any(col.startswith("20_R") for col in cols):
            return "1X1"
        if any(col.startswith("1_R") for col in cols) and any(col.startswith("16_R") for col in cols):
            return "4X4"
        gas_cols = {"TEMP", "HUM", "PRESS"} & set(cols)
        if gas_cols and any("TGS" in col or col.startswith("NE4") or col.startswith("SS") or col.startswith("MINIPID") for col in cols):
            return "GAS"
    return None


def normalize_folder(folder: Path, dry_run=True):
    if not folder.is_dir():
        return []

    messages = []
    for csv_path in sorted(folder.glob("*.csv")):
        target = infer_type(csv_path)
        if target is None:
            continue
        new_path = folder / f"{target}.csv"
        if csv_path.name == new_path.name:
            continue
        if new_path.exists():
            messages.append((csv_path, None, f"skip: target exists: {new_path.name}"))
            continue

        if dry_run:
            messages.append((csv_path, new_path, "dry-run"))
        else:
            csv_path.rename(new_path)
            messages.append((csv_path, new_path, "renamed"))
    return messages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize sensor CSV file names to 1X1.csv, 4X4.csv, and GAS.csv.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR, help="Root data directory")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be renamed without changing files")
    args = parser.parse_args()

    root = args.data_dir
    if not root.exists():
        raise SystemExit(f"data directory not found: {root}")

    all_msgs = []
    for session in sorted(root.glob("**")):
        if not session.is_dir():
            continue
        msgs = normalize_folder(session, dry_run=args.dry_run)
        all_msgs.extend([(session, *m) for m in msgs])

    if not all_msgs:
        print("No files detected for renaming.")
    else:
        for session, old, new, status in all_msgs:
            print(f"{session}: {old.name} -> {new.name if new else 'N/A'} [{status}]")
        if args.dry_run:
            print(f"\nDry run complete. {len(all_msgs)} changes would be made.")
        else:
            print(f"\nRenamed {len(all_msgs)} files.")
