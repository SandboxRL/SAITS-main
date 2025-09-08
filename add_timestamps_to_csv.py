import pandas as pd
from pathlib import Path
import vitaldb as vdb

# Track names must match what you used when you trained SAITS
TRACK_KEEP = ["SNUADC/ART ", "SNUADC/ECG_II", "SNUADC/ECG_V5 ",
              "SNUADC/PLETH", "Primus/CO2", "BIS/EEG1_WAV", "BIS/EEG2_WAV"]

# Root directory
ROOT = Path("~/vitaldb_bulk/physionet.org/vital_files").expanduser()

def get_timestamp_index(vital_path: Path):
    """Returns datetime index from .vital file."""
    df = vdb.vital_recs(
        str(vital_path),
        track_names=TRACK_KEEP,
        return_timestamp=False,
        return_datetime=True,     # crucial for real timestamps
        return_pandas=True,
    )
    return df.index[:len(df)]    # align index with same length as df

def patch_csv_with_timestamp(csv_path: Path):
    stem_without_imputed = csv_path.stem.replace("_imputed", "")
    vital_path = csv_path.with_name(stem_without_imputed + ".vital")

    if not vital_path.exists():
        print(f"❌ No matching .vital for {csv_path.name}")
        return

    # Load imputed CSV
    try:
        df_csv = pd.read_csv(csv_path)
        # Skip if 'timestamp' already exists
        if "timestamp" in df_csv.columns:
            print(f"⏭️  Skipping {csv_path.name} (already has timestamp)")
            return
        ts_index = get_timestamp_index(vital_path)
        
        if len(df_csv) > len(ts_index):
            print(f"⚠️  CSV has MORE rows than VitalDB file ({csv_path.name}); skipping.")
            return
        elif len(df_csv) < len(ts_index):
            print(f"ℹ️  Trimming timestamps to match CSV rows for {csv_path.name}")
            ts_index = ts_index[:len(df_csv)]  # take only as many as needed

        df_csv.insert(0, "timestamp", ts_index)
        df_csv.to_csv(csv_path, index=False)
        print(f"✅ Added timestamps to {csv_path.name}")
    except Exception as e:
        print(f"❌ Failed on {csv_path.name}: {e}")

def main():
    csv_files = sorted(ROOT.rglob("*.csv"))
    for csv in csv_files:
        patch_csv_with_timestamp(csv)

if __name__ == "__main__":
    main()
