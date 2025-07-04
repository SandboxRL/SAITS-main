import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, calc_missing_rate
from benchpots.datasets import preprocess_physionet2012

## Load & resample all VitalDB files

from pathlib import Path
import os, vitaldb as vdb, numpy as np, pandas as pd

# ---------- settings ----------
#vital_dir      = Path("/Users/muhammadaneequz.zaman/Dropbox/Digital Twin (Umer Huzaifa)/vitalDB_v1")
# directory that contains this .py file
script_dir = Path(__file__).resolve().parent
track_keep     = ["SNUADC/ART ", "SNUADC/ECG_II", "SNUADC/ECG_V5 ", 
                  "SNUADC/PLETH", "Primus/CO2", "BIS/EEG1_WAV", "BIS/EEG2_WAV"]   # pick tracks present in *every* case
target_fs      = "1S"                                           # 1-second grid
sequence_len   = 60*10                                          # 10-min snippets → n_steps = 600
# ------------------------------

def read_one(file_path, tracks_keep):
    """
    Return a pandas DataFrame whose index is the native VitalDB timestamp
    (datetime) and whose columns are the requested track names.
    Missing samples remain as NaN.
    """
    all_tracks = vdb.vital_trks(str(file_path))
    #print(all_tracks)
    print(tracks_keep)
    #numeric_tracks = [t for t in all_tracks if t in tracks_keep]

    # if not numeric_tracks:
    #     raise ValueError("none of the requested tracks in this file")
    
    return vdb.vital_recs(
        str(file_path),
        # track_names=all_tracks,
        track_names=tracks_keep,
        return_timestamp=False,      # keep absolute clock time
        return_datetime=False,
        return_pandas=True,
    )

all_cases = []                      # dict: filename  -> DataFrame
bad_files = []
#for f in sorted(vital_dir.glob("*.vital")):
for f in sorted(script_dir.glob("*.vital")):
    try:
        df = read_one(f, track_keep)
        df_numeric = df.apply(pd.to_numeric, errors="coerce")  # strings -> NaN
        all_cases.append(df_numeric.to_numpy(dtype=np.float32))   # shape (sequence_len, n_features)
    except Exception as e:
        print(f"skip {f.name}: {e}")
        bad_files.append(f.name)

dataset = np.stack(all_cases, axis=0)               # ==> (n_samples, n_steps, n_features)
print("Dataset shape:", dataset.shape, "  (skipped", len(bad_files), "files)")

## Train / val / test split

from sklearn.model_selection import train_test_split

X            = dataset.squeeze(0)          # → (5_771_049, 80)
window       = 600                         # 10 minutes if you resample to 1 Hz
stride       = 600                         # non-overlapping; use <window for overlap
segments = [
    X[i : i + window]
    for i in range(0, X.shape[0] - window + 1, stride)
]
segments = np.stack(segments)              # (n_segments, 600, 80)
print("segments shape:", segments.shape)

train_X, test_X = train_test_split(segments,  test_size=0.15, random_state=42)
train_X,  val_X = train_test_split(train_X, test_size=0.15, random_state=42)

print("train", train_X.shape, "val", val_X.shape, "test", test_X.shape)

## Add extra synthetic missingness on val set

from pygrinder import mcar, calc_missing_rate

val_X_ori = val_X.copy()             # keep a pristine copy
val_X     = mcar(val_X, p=0.10)   # mask-at-random 10 %

test_X_ori = test_X.copy()           # ditto for the test set
indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)
print(f"Real miss rate train  : {calc_missing_rate(train_X):.1%}")
print(f"Real+fake miss rate val: {calc_missing_rate(val_X):.1%}")

## Wrap in the dictionaries SAITS expects

train_set = {"X": train_X}
val_set   = {"X": val_X, "X_ori": val_X_ori}
test_set  = {"X": test_X}

## Instantiate, train, evaluate just like the example

from pypots.imputation import SAITS
from pypots.nn.functional import calc_mae

saits = SAITS(
    n_steps   = train_X.shape[1],
    n_features= train_X.shape[2],
    n_layers  = 2,
    d_model   = 256,
    n_heads   = 4,
    d_k       = 64,
    d_v       = 64,
    d_ffn     = 128,
    dropout   = 0.1,
    epochs    = 200,
    patience  = 5,                  # early-stop patience (optional)
    device    = "cpu"
    #device    = "cuda:0"
)

saits.fit(train_set, val_set)

# ---- test-time imputation ----
imputation = saits.impute(test_set)               # same shape as test_X
mae        = calc_mae(imputation, np.nan_to_num(test_X_ori), indicating_mask)
print("MAE on held-out values:", mae)

saits.save("models/saits_vitaldb.pypots", overwrite=True)
saits.load("models/saits_vitaldb.pypots")

## IMPUTE THE ORIGINAL DATA  (train + val + test, NaNs only)

# Concatenate all three splits so we fill every real gap in one go
orig_concat = np.concatenate([train_X, val_X_ori, test_X_ori], axis=0)
orig_imputed = saits.impute({"X": orig_concat})        # <-- returns np.ndarray

# You can now split it back if you want
n_train = train_X.shape[0]
n_val   = val_X_ori.shape[0]
imputed_train = orig_imputed[:n_train]
imputed_val_full = orig_imputed[n_train:n_train+n_val]  # val set with real NaNs filled
imputed_test  = orig_imputed[n_train+n_val:]

## IMPUTE THE SYNTHETICALLY MASKED VALIDATION SET

# val_X has *extra* 10 % MCAR holes; we already built val_set = {"X": val_X}
imputed_val_masked = saits.impute(val_set)              # same shape as val_X
# evaluate MAE on those artificial holes
masked_mae = calc_mae(imputed_val_masked, 
                      np.nan_to_num(val_X_ori), 
                      np.isnan(val_X) ^ np.isnan(val_X_ori))
print("MAE on synthetically missing points in val set:", masked_mae)

## SHOW 15 RANDOM IMPUTATIONS vs. GROUND-TRUTH

import random, pandas as pd

feature_names = track_keep                        # your 7 channels in that order
mask_idx = np.where((np.isnan(val_X)) & ~np.isnan(val_X_ori))  # positions you hid
n_show = min(30, mask_idx[0].size)               # show up to 15 rows
rows = random.sample(range(mask_idx[0].size), n_show)

print(mask_idx)
records = []
for k in rows:
    s, t, f = mask_idx[0][k], mask_idx[1][k], mask_idx[2][k]
    records.append({
        "sample#":    s,
        "time_step":  t,
        "channel":    feature_names[f],
        "ground_truth": float(val_X_ori[s, t, f]),
        "imputed":     float(imputed_val_masked[s, t, f]),
        "abs_error":   abs(val_X_ori[s, t, f] - imputed_val_masked[s, t, f]),
    })

comparison_df = pd.DataFrame(records)
print("\n===  SAITS imputation on synthetic holes (random 30)  ===")
print(comparison_df.round(3).to_string(index=False))