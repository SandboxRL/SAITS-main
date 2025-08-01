# train_and_impute_one.py
# ---------------------------------------
import os, sys, random, numpy as np, pandas as pd
from pathlib import Path
import vitaldb as vdb
from sklearn.model_selection import train_test_split
from pygrinder import mcar, calc_missing_rate
from pypots.imputation import SAITS
from pypots.nn.functional import calc_mae

# ---------- USER SETTINGS ----------
TRACK_KEEP   = ["SNUADC/ART ", "SNUADC/ECG_II", "SNUADC/ECG_V5 ",
                "SNUADC/PLETH", "Primus/CO2", "BIS/EEG1_WAV", "BIS/EEG2_WAV"]
WINDOW       = 600      # 10 min at 1-Hz grid
TARGET_FS    = "1S"
EPOCHS       = 200      # 2-3 min per file on A100; raise if needed
DEVICE       = "cuda:0"
# ------------------------------------

def read_one(file_path: Path) -> np.ndarray:
    """Return (n_steps, n_features) array for one .vital file."""
    df = vdb.vital_recs(
        str(file_path),
        track_names=TRACK_KEEP,
        return_timestamp=False,       # keep relative sampling
        return_datetime=False,
        return_pandas=True,
    ).apply(pd.to_numeric, errors="coerce")

    return df.to_numpy(dtype=np.float32)

def windows(x: np.ndarray, win: int = WINDOW, stride: int = WINDOW):
    return np.stack([x[i:i+win] for i in range(0, len(x)-win+1, stride)])

def train_and_impute(arr: np.ndarray, out_prefix: Path):
    """Train SAITS on windows of arr and save imputed CSV + weights."""
    # --- window -> (samples, steps, features) ---
    seg = windows(arr)                      # shape (n_seg, 600, 7)

    # --- split ---
    tr, te = train_test_split(seg, test_size=0.15, random_state=42)
    tr, va = train_test_split(tr, test_size=0.15, random_state=42)

    # synthetic missingness on val
    va_ori = va.copy()
    va = mcar(va, p=0.10)

    # SAITS
    saits = SAITS(
        n_steps   = tr.shape[1],
        n_features= tr.shape[2],
        n_layers  = 2, d_model=256, n_heads=4,
        d_k=64, d_v=64, d_ffn=128,
        dropout=0.1, epochs=EPOCHS, device=DEVICE)
    saits.fit({"X": tr}, {"X": va, "X_ori": va_ori})

    # --- full imputation ---
    imputed = saits.impute({"X": seg})      # ndarray (n_seg, 600, 7)

    # reshape back into one long series
    flat = imputed.reshape(-1, imputed.shape[-1])[:len(arr)]
    df_imp = pd.DataFrame(flat, columns=TRACK_KEEP)
    df_imp.to_csv(f"{out_prefix}_imputed.csv", index=False)

    saits.save(f"{out_prefix}_saits.pypots", overwrite=True)

    print(f"✔ saved {out_prefix}_imputed.csv")

def main():
    root = Path("~/vitaldb_bulk/physionet.org/vital_files").expanduser()
    files = sorted(root.rglob("*.vital"))
    if not files:
        sys.exit("No .vital files found!")

    for f in files:
        out_prefix = f.with_suffix("")      # e.g. 1234/1234_000
        if (out_prefix.parent / f"{out_prefix.stem}_imputed.csv").exists():
            print(f"skip {f.name} (already done)")
            continue
        try:
            data = read_one(f)
            train_and_impute(data, out_prefix)
        except Exception as e:
            print(f"❌ {f.name}: {e}")

if __name__ == "__main__":
    main()
