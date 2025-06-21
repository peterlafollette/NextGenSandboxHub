###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# helps inspect runs as they are in progress, showing the latest iteration and best calibration objective function value so far. Also shows the NWM objective function value for the validation period.
# won't show validation metrics as the run progresses because the calibration iterations will not run over the validation period. Will show validaiton metrics once calibration has finished for a gage.


import os
import pandas as pd

# === ANSI Colors ===
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# === USER SETTINGS ===
objective_metric = "kge"   # <<< Set to "kge", "log_kge", "event_kge", etc, although might be the case that only KGE is supported now.

# === Path setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

logging_dir = os.path.join(project_root, "logging")
nwm_results_path = os.path.join(project_root, "model_assessment", "retro_assessment", "KGE_results.csv")

# === Load NWM validation KGEs ===
nwm_df = pd.read_csv(nwm_results_path, dtype={"gage_id": str})
nwm_kge_lookup = nwm_df.set_index("gage_id")[["KGE", "log_KGE"]].to_dict(orient="index")

# === Final summary ===
summary = []

# === Process each gage log ===
for filename in sorted(os.listdir(logging_dir)):
    if not filename.endswith(".csv"):
        continue

    gage_id = filename.replace(".csv", "")
    filepath = os.path.join(logging_dir, filename)

    try:
        df = pd.read_csv(filepath)
        df["iteration_str"] = df["iteration"].astype(str)

        # === Show latest numeric iteration for info (not used for metrics) ===
        df_numeric = df[pd.to_numeric(df["iteration_str"], errors="coerce").notna()].copy()
        df_numeric["iteration"] = df_numeric["iteration_str"].astype(int)
        if df_numeric.empty:
            print(f"\n=== Gage {gage_id} ===")
            print(" No numeric iterations found.")
            continue

        latest_iter = df_numeric["iteration"].max()
        latest_df = df[df["iteration_str"] == str(latest_iter)]

        print(f"\n=== Gage {gage_id} | Latest Iteration: {latest_iter} ===")
        for _, row in latest_df.iterrows():
            particle = int(row["particle"])
            cal_metric = float(row.get(f"{objective_metric}_calibration", float("nan")))
            val_metric = float(row.get(f"{objective_metric}_validation", float("nan")))
            print(f"Particle {particle:02d} | {objective_metric}_cal: {cal_metric:.4f} | {objective_metric}_val: {val_metric:.4f}")

        # === Use final row only ===
        final_row = df[df["iteration_str"].str.lower() == "final"]
        if final_row.empty:
            print(f" No 'final' row found in {filename}; skipping.")
            continue

        best_row = final_row.iloc[0]
        best_particle = int(best_row["particle"])
        best_cal = float(best_row[f"{objective_metric}_calibration"])
        best_val = float(best_row[f"{objective_metric}_validation"])

        print(f"\n Best Particle: {best_particle:02d} from 'final'")
        print(f"{objective_metric}_cal: {best_cal:.4f} | {objective_metric}_val: {best_val:.4f}")

        # === Compare to NWM ===
        nwm_val = None
        if objective_metric == "kge" and gage_id in nwm_kge_lookup:
            nwm_val = nwm_kge_lookup[gage_id]["KGE"]
        elif objective_metric == "log_kge" and gage_id in nwm_kge_lookup:
            nwm_val = nwm_kge_lookup[gage_id]["log_KGE"]

        if nwm_val is not None:
            print(f"\n NWM Retrospective {objective_metric}_val: {nwm_val:.4f}")

            if best_val > nwm_val:
                verdict = " Calibrated model performed better (Validation)"
                comparison = f"{GREEN}Calibrated model{RESET}"
            elif best_val < nwm_val:
                verdict = "  Retrospective NWM performed better (Validation)"
                comparison = f"{YELLOW}Retrospective{RESET}"
            else:
                verdict = " Tie between LGAR and NWM (Validation)"
                comparison = "Tie"
        else:
            verdict = " No direct NWM comparison for this metric"
            comparison = "Missing"
            nwm_val = None

        print(verdict)

        summary.append({
            "gage_id": gage_id,
            f"model_{objective_metric}_cal": best_cal,
            f"model_{objective_metric}_val": best_val,
            "nwm_val": nwm_val,
            "comparison": comparison,
            "latest_iter": latest_iter
        })

    except Exception as e:
        print(f" Error reading {filename}: {e}")

# === Final summary ===
print("\n\n=== Retrospective vs Calibrated Model Summary ===")
print(f"{'Gage':<12} | {objective_metric+'_cal':<12} | {objective_metric+'_val':<12} | {'NWM':<10} | {'Preference':<21} | {'Latest Iter':<12}")
print("-" * 90)

for entry in sorted(summary, key=lambda x: x["gage_id"]):
    nwm_val = entry["nwm_val"]
    nwm_val_str = f"{nwm_val:.4f}" if isinstance(nwm_val, (float, int)) and not pd.isna(nwm_val) else "-"
    
    print(
        f"{entry['gage_id']:<12} | "
        f"{entry[f'model_{objective_metric}_cal']:<12.4f} | "
        f"{entry[f'model_{objective_metric}_val']:<12.4f} | "
        f"{nwm_val_str:<10} | "
        f"{entry['comparison']:<30} | "
        f"{entry['latest_iter']:<12}"
    )

# === Compute fraction of sites where LGAR-based model was preferred ===
num_total = len(summary)
num_lgar_better = sum("Calibrated model" in s["comparison"] for s in summary)
fraction = num_lgar_better / num_total if num_total > 0 else 0

print("\n" + "=" * 80)
print(f"Calibrated model was preferred at {num_lgar_better} of {num_total} sites ({fraction:.1%})")
