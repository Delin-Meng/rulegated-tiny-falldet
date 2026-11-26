"""Plot acceleration of one SA06 trial (S06T20R01, fall F01) to check data loading."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

ROOT = cfg.DATASET_DIR / "KFall"
CSV = ROOT / "sensor_data_new" / "SA06" / "S06T20R01.csv"
LABEL = ROOT / "label_data_new" / "SA06_label.xlsx"
OUT = cfg.RESULTS_DIR / "SA06_acceleration.png"

df = pd.read_csv(CSV)

# Task code is written only on the first row of each task, so fill it downward
lab = pd.read_excel(LABEL)
lab["Task Code (Task ID)"] = lab["Task Code (Task ID)"].ffill()
row = lab[(lab["Task Code (Task ID)"] == "F01 (20)") & (lab["Trial ID"] == 1)].iloc[0]
onset, impact = int(row["Fall_onset_frame"]), int(row["Fall_impact_frame"])

# Label frame number -> time (s), matched via the FrameCounter column
t_onset = df.loc[df["FrameCounter"] == onset, "TimeStamp(s)"].iloc[0]
t_impact = df.loc[df["FrameCounter"] == impact, "TimeStamp(s)"].iloc[0]

t = df["TimeStamp(s)"]
svm = np.sqrt(df["AccX"] ** 2 + df["AccY"] ** 2 + df["AccZ"] ** 2)  # acceleration magnitude

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t, df["AccX"], label="AccX", lw=1)
ax.plot(t, df["AccY"], label="AccY", lw=1)
ax.plot(t, df["AccZ"], label="AccZ", lw=1)
ax.plot(t, svm, label="|a| (magnitude)", lw=1.4, color="black")
ax.axvline(t_onset, color="tab:orange", ls="--", label=f"Fall onset (frame {onset})")
ax.axvline(t_impact, color="tab:red", ls="--", label=f"Impact (frame {impact})")
ax.set_title(f"SA06 / {CSV.name} (F01 forward fall when trying to sit down) - acceleration, {len(df)} rows @100Hz")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (g)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", ncol=3, fontsize=9)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print("Saved:", OUT)
print("Rows:", len(df), "| onset", onset, "-> t", t_onset, "| impact", impact, "-> t", t_impact)
