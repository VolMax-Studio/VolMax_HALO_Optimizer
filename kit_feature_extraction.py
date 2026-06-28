import os
import zipfile
import pandas as pd
import numpy as np

# Paths
base_dir = "/home/volmax-studio/volmax-projects/iot2/PORTFOLIO/VolMax_HALO_Optimizer/kit_result_data/10.35097-1969/data/dataset"
cfg_zip_path = os.path.join(base_dir, "cfg.zip")
eoc_zip_path = os.path.join(base_dir, "cell_eocv2.zip")
pls_zip_path = os.path.join(base_dir, "cell_plsv2.zip")
output_csv = "/home/volmax-studio/volmax-projects/iot2/PORTFOLIO/VolMax_HALO_Optimizer/kit_features.csv"

# 1. Load cell configurations for cyclic aging cells
print("Loading configurations...")
cyclic_cells = {}
with zipfile.ZipFile(cfg_zip_path, 'r') as z:
    for name in z.namelist():
        if name.endswith('.csv') and 'cell_cfg_' in name:
            with z.open(name) as f_in:
                df = pd.read_csv(f_in, sep=";")
                if len(df) > 0 and df.iloc[0]["age_type"] == 2:
                    basename = os.path.basename(name)
                    parts = basename.replace("cell_cfg_", "").replace(".csv", "").split("_")
                    cell_name = f"{parts[0]}_{parts[1]}"
                    cyclic_cells[cell_name] = {
                        "temp": float(df.iloc[0]["age_temp"]),
                        "chg": float(df.iloc[0]["age_chg_rate"]),
                        "dischg": float(df.iloc[0]["age_dischg_rate"]),
                        "instance": int(parts[1])
                    }

print(f"Found {len(cyclic_cells)} cyclic cells.")

# 2. Extract features and targets
features_list = []

# Helper for linear slope
def get_slope(x, y):
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])

with zipfile.ZipFile(eoc_zip_path, 'r') as z_eoc, zipfile.ZipFile(pls_zip_path, 'r') as z_pls:
    eoc_names = z_eoc.namelist()
    pls_names = z_pls.namelist()
    
    for idx, (cell_name, info) in enumerate(cyclic_cells.items()):
        # Find matching zip files
        eoc_matching = [f for f in eoc_names if cell_name in f]
        pls_matching = [f for f in pls_names if cell_name in f]
        
        if not eoc_matching or not pls_matching:
            print(f"Skipping {cell_name} - EOC or PLS file not found.")
            continue
            
        # Read EOC (Capacity, Temp)
        with z_eoc.open(eoc_matching[0]) as f_eoc:
            df_eoc = pd.read_csv(f_eoc, sep=";")
        
        # Read PLS (Internal Resistance R0/R1)
        with z_pls.open(pls_matching[0]) as f_pls:
            df_pls = pd.read_csv(f_pls, sep=";")
            
        # Clean EOC data
        df_eoc_clean = df_eoc.dropna(subset=["cap_aged_est_Ah", "soh_cap"]).copy()
        if len(df_eoc_clean) == 0:
            print(f"Skipping {cell_name} - EOC clean data empty.")
            continue
            
        # Add EFC proxy (nominal capacity proxy is 2.9 Ah)
        df_eoc_clean["EFC"] = df_eoc_clean["total_q_chg_sum_Ah"] / 2.9
        
        # Find cycle_life (EFC where SOH falls below 80%)
        below_80 = df_eoc_clean[df_eoc_clean["soh_cap"] <= 80.0]
        if len(below_80) > 0:
            target_cycle_life = float(below_80.iloc[0]["EFC"])
        else:
            # Fallback if 80% is not strictly reached, use extrapolation or max EFC
            target_cycle_life = float(df_eoc_clean["EFC"].max())
            
        # Get early-cycle EOC data (EFC <= 10)
        df_eoc_early = df_eoc_clean[df_eoc_clean["EFC"] <= 10.0].copy()
        if len(df_eoc_early) < 2:
            print(f"Skipping {cell_name} - Too few early EOC points (len={len(df_eoc_early)}).")
            continue
            
        # Extract EOC early features
        cap_initial = float(df_eoc_early.iloc[0]["cap_aged_est_Ah"])
        cap_final = float(df_eoc_early.iloc[-1]["cap_aged_est_Ah"])
        cap_diff = cap_final - cap_initial
        cap_slope = get_slope(df_eoc_early["EFC"].values, df_eoc_early["cap_aged_est_Ah"].values)
        
        temp_vals = df_eoc_early["t_start_degC"].dropna()
        if len(temp_vals) > 0:
            temp_avg = float(temp_vals.mean())
            temp_slope = get_slope(np.arange(len(temp_vals)), temp_vals.values)
        else:
            temp_avg = info["temp"]
            temp_slope = 0.0
            
        # Filter PLS for early cycles (timestamp_s <= max timestamp of early EOC)
        max_early_time = df_eoc_early["timestamp_s"].max()
        df_pls_early = df_pls[df_pls["timestamp_s"] <= max_early_time].dropna(subset=["r_ref_10ms_mOhm"]).copy()
        
        # If no PLS early data, try check-ups associated with EFC <= 10 using block IDs or timestamps
        if len(df_pls_early) == 0:
            # Fallback: take first 100 rows or similar if timestamps are offset
            df_pls_early = df_pls.dropna(subset=["r_ref_10ms_mOhm"]).head(100).copy()
            
        if len(df_pls_early) < 2:
            r0_initial = 25.0 # fallback values in mOhm
            r0_diff = 0.0
            r0_slope = 0.0
        else:
            r0_initial = float(df_pls_early.iloc[0]["r_ref_10ms_mOhm"])
            r0_final = float(df_pls_early.iloc[-1]["r_ref_10ms_mOhm"])
            r0_diff = r0_final - r0_initial
            # fit slope against timestamp (scaled to days)
            time_days = (df_pls_early["timestamp_s"].values - df_pls_early["timestamp_s"].values[0]) / 86400.0
            r0_slope = get_slope(time_days, df_pls_early["r_ref_10ms_mOhm"].values)
            
        features_list.append({
            "cell_name": cell_name,
            "temp": info["temp"],
            "chg_rate": info["chg"],
            "dischg_rate": info["dischg"],
            "instance": info["instance"],
            "cap_initial": cap_initial,
            "cap_diff": cap_diff,
            "cap_slope": cap_slope,
            "temp_avg": temp_avg,
            "temp_slope": temp_slope,
            "r0_initial": r0_initial,
            "r0_diff": r0_diff,
            "r0_slope": r0_slope,
            "target_cycle_life": target_cycle_life
        })
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(cyclic_cells)} cells...")

# 3. Create DataFrame and export
df_features = pd.DataFrame(features_list)
df_features.to_csv(output_csv, index=False)
print(f"\nFeature extraction complete. Exported {len(df_features)} cells to {output_csv}")
print(df_features.head())
print("\nTarget Cycle Life Stats:")
print(df_features["target_cycle_life"].describe())
