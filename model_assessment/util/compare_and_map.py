###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | May 2025]
# plots a map indicating at each gage which model did best, makes box plots describing model performance among all sites with various error metrics, and then makes a table describing the impact of removing each individual model on ensemble skill

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import box
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_assessment.util.expanded_metrics import compute_metrics
from model_assessment.configs import path_config as cfg
project_root = cfg.project_root
import yaml

# === USER SETTINGS ===
objective_metric = "kge"   # or "log_kge" or "event_kge", although these may no lojnger be supported 
exclude_low     = False    # whether to filter out winners below threshold
kge_threshold   = 0.3      # threshold if exclude_low=True
plot_hydrographs = True    # Set to False to skip hydrograph plots
annotate = True            #prints gage IDs on the map

# === Paths ===
nwm_results_path = os.path.join(
    cfg.project_root,
    "model_assessment", "retro_assessment", "kge_results.csv"
)
gage_to_comid_path = os.path.join(
    cfg.model_assessment_root, "retro_assessment", "get_COMIDS", "gage_comid_from_nhdplus_national.csv"
)
comid_info_path = os.path.join(
    cfg.model_assessment_root, "retro_assessment", "get_COMIDS", "gage_comid_full_nhdplus_national.csv"
)

###this is just a shapefile I use for making a map, it doesn't have to be this one in particular
###it is from https://catalog.data.gov/dataset/2023-cartographic-boundary-file-shp-state-and-equivalent-entities-for-united-states-1-20000000/resource/b5ec957e-2ba3-4f3f-b0a3-c01973a50aec  
states_shapefile   = os.path.join(
    cfg.model_assessment_root, "cb_2023_us_state_20m", "cb_2023_us_state_20m.shp"
)
retro_q_dir = os.path.join(
    cfg.model_assessment_root, "retro_assessment", "retro_q_sims"
)
obs_dir = os.path.join(
    cfg.model_assessment_root, "USGS_streamflow", "successful_sites_resampled"
)

winner_csv_path    = "gage_model_winners.csv"

lasam_log_dir = os.path.join(project_root, "model_assessment", "output_for_visualization", "lasam", "logging")
cfe_log_dir   = os.path.join(project_root, "model_assessment", "output_for_visualization", "cfe", "logging")

# === Load NWM KGE lookup ===
nwm_df = pd.read_csv(nwm_results_path, dtype={"gage_id":str})
nwm_kge_lookup = nwm_df.set_index("gage_id")[["KGE","log_KGE"]].to_dict(orient="index")

# === Load gage-to-COMID mapping ===
g2c_df = pd.read_csv(gage_to_comid_path, dtype={'gage_id':str,'COMID':str})
gage_to_comid = dict(zip(g2c_df['gage_id'], g2c_df['COMID']))

# === Helper: load NWM retrospective series ===
def load_nwm_series(comid):
    path = os.path.join(retro_q_dir, f"{comid}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
    return df.set_index('time')['streamflow']

def extract_best_val_metric(log_dir, gage_id):
    fp = os.path.join(log_dir, f"{gage_id}.csv")
    if not os.path.exists(fp):
        return None

    df = pd.read_csv(fp)

    # If 'final' row exists, use that directly
    if 'iteration' in df.columns:
        final_row = df[df['iteration'].astype(str).str.lower() == 'final']
        if not final_row.empty:
            row = final_row.iloc[0]
            return {
                'cal': float(row.get(f"{objective_metric}_calibration", np.nan)),
                'val': float(row.get(f"{objective_metric}_validation", np.nan))
            }

    # Fallback: use max calibration score from numeric iterations. Will not work with updated code that only runs the model over the calibration period during calibration
    df_clean = df[df['iteration'].astype(str).str.isdigit()].copy()
    if df_clean.empty:
        return None
    df_clean['iteration'] = df_clean['iteration'].astype(int)
    best = df_clean.loc[df_clean[f"{objective_metric}_calibration"].idxmax()]
    return {
        'cal': float(best[f"{objective_metric}_calibration"]),
        'val': float(best[f"{objective_metric}_validation"])
    }


# === Gather gage IDs and determine winners ===
gage_ids = sorted(
    {f.replace('.csv','') for f in os.listdir(lasam_log_dir)
     if f.endswith('.csv') and not f.startswith('._')} |
    {f.replace('.csv','') for f in os.listdir(cfe_log_dir)
     if f.endswith('.csv') and not f.startswith('._')}
)

summary = []
for gid in gage_ids:
    las = extract_best_val_metric(lasam_log_dir, gid)
    cfe = extract_best_val_metric(cfe_log_dir,   gid)
    nwm_val = (
        nwm_kge_lookup.get(gid,{}).get('KGE')
        if objective_metric=='kge'
        else nwm_kge_lookup.get(gid,{}).get('log_KGE')
    )
    if not las or not cfe or nwm_val is None:
        continue
    scores = {'LGAR - PF': las['val'], 'CFE': cfe['val'], 'NWM retro': nwm_val}
    winner = max(scores, key=scores.get)
    summary.append({
        'gage_id':   gid,
        'lasam_val': las['val'],
        'cfe_val':   cfe['val'],
        'nwm_val':   nwm_val,
        'winner':    winner
    })


summary_df = pd.DataFrame(summary).drop_duplicates(subset='gage_id')

# === Optional filter of low winners ===
if exclude_low:
    winner_to_col = {
        'LGAR - PF': 'lasam_val',
        'CFE': 'cfe_val',
        'NWM retro': 'nwm_val'
    }
    mask = summary_df.apply(lambda r: r[winner_to_col[r['winner']]] >= kge_threshold, axis=1)
    summary_df = summary_df[mask]

summary_df.to_csv(winner_csv_path, index=False, columns=['gage_id','winner'])
counts = summary_df['winner'].value_counts().to_dict()


# summary_df = pd.DataFrame(summary).drop_duplicates(subset='gage_id')
summary_df.to_csv(winner_csv_path, index=False, columns=['gage_id','winner'])
counts = summary_df['winner'].value_counts().to_dict()
print("Preference:", counts)

# === Color terminal output ===
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# # === Print color-coded per-gage performance table ===
# print("\n=== Validation Metric Comparison ===")
# print(f"{'Gage':<12} | {'LGAR - PF':<10} | {'CFE':<10} | {'NWM':<10} | Best")
# print("-" * 60)
# for row in summary:
#     gid = row['gage_id']
#     las = row['lasam_val']
#     cfe = row['cfe_val']
#     nwm = row['nwm_val']
#     win = row['winner']
#     color = {
#         'LGAR - PF': GREEN,
#         'CFE': BLUE,
#         'NWM retro': YELLOW
#     }.get(win, "")
#     print(f"{gid:<12} | {las:<10.4f} | {cfe:<10.4f} | {nwm:<10.4f} | {color}{win}{RESET}")

print("\n=== Validation Metric Comparison ===")
print(f"{'Gage':<12} | {'LGAR - PF':<10} | {'CFE':<10} | {'NWM':<10} | Best")
print("-" * 60)
for _, row in summary_df.iterrows():
    gid  = row['gage_id']
    las  = row['lasam_val']
    cfe  = row['cfe_val']
    nwm  = row['nwm_val']
    win  = row['winner']

    color = {
        'LGAR - PF': GREEN,
        'CFE': BLUE,
        'NWM retro': YELLOW
    }.get(win, "")

    nwm_str = f"{nwm:<10.4f}" if not np.isnan(nwm) else " " * 10
    print(f"{gid:<12} | {las:<10.4f} | {cfe:<10.4f} | {nwm_str} | {color}{win}{RESET}")



# === Optional filter of low winners ===
if exclude_low:
    winner_to_col = {
        'LGAR - PF': 'lasam_val',
        'CFE': 'cfe_val',
        'NWM retro': 'nwm_val'
    }
    mask = summary_df.apply(lambda r: r[winner_to_col[r['winner']]] >= kge_threshold, axis=1)
    summary_df = summary_df[mask]
    summary_df.to_csv(winner_csv_path, index=False, columns=['gage_id','winner'])
    counts = summary_df['winner'].value_counts().to_dict()
    print(" Filtered:", counts)


# === Compute error metrics ===
metrics = ['volume_error_percent','peak_flow_error_percent','time_to_peak_error_hours']
labels = ['LGAR - PF','CFE','NWM retro','Best Overall']
colors = {'LGAR - PF':'green','CFE':'blue','NWM retro':'gold','Best Overall':'cyan'}

# Initialize container
error = {lbl: {m:[] for m in metrics} for lbl in labels}

# Validation window
# start_time, end_time = '2018-10-01','2020-09-30' ###using time config now
with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

cal_start = pd.Timestamp(time_cfg["cal_start"])
cal_end   = pd.Timestamp(time_cfg["cal_end"])
val_start = pd.Timestamp(time_cfg["val_start"])
val_end   = pd.Timestamp(time_cfg["val_end"])

start_time, end_time = val_start, val_end

for idx, row in summary_df.iterrows():
    gid, win = row['gage_id'], row['winner']
    # Observed
    obs_fp = os.path.join(obs_dir, f"{gid}.csv")
    obs_df = pd.read_csv(obs_fp, parse_dates=['value_time']).set_index('value_time')['flow_m3_per_s']
    # LASAM
    las_pp = lasam_log_dir.replace('logging','postproc')
    las_df = pd.read_csv(os.path.join(las_pp, f"{gid}_best.csv"),
                         parse_dates=['current_time']).set_index('current_time')['flow']
    # CFE
    cfe_pp = cfe_log_dir.replace('logging','postproc')
    cfe_df = pd.read_csv(os.path.join(cfe_pp, f"{gid}_best.csv"),
                         parse_dates=['current_time']).set_index('current_time')['flow']
    # NWM retro
    comid = gage_to_comid.get(gid)
    nwm_series = load_nwm_series(comid) if comid else None

    # Compute metrics
    results = {}
    results['LGAR - PF'] = compute_metrics(las_df, obs_df, start_time=start_time, end_time=end_time)
    results['CFE']       = compute_metrics(cfe_df, obs_df, start_time=start_time, end_time=end_time)
    if nwm_series is not None:
        results['NWM retro'] = compute_metrics(nwm_series, obs_df, start_time=start_time, end_time=end_time)
    else:
        results['NWM retro'] = {m: np.nan for m in metrics}

    # Record per-model errors
    for lbl in ['LGAR - PF','CFE','NWM retro']:
        for m in metrics:
            val = abs(results[lbl][m]) if 'percent' in m else results[lbl][m]
            error[lbl][m].append(val)

    # Best Overall
    for m in metrics:
        if win in results:
            best_val = abs(results[win][m]) if 'percent' in m else results[win][m]
        else:
            best_val = np.nan
        error['Best Overall'][m].append(best_val)

# === Load geometries for mapping ===
gage_df   = pd.read_csv(gage_to_comid_path, dtype={'gage_id':str,'COMID':str})
comid_geo = pd.read_csv(comid_info_path)[['FLComID','geometry']].rename(columns={'FLComID':'COMID'})
comid_geo['COMID'] = comid_geo['COMID'].astype(str)
comid_geo['geometry'] = comid_geo['geometry'].apply(wkt.loads)
gdf_com   = gpd.GeoDataFrame(comid_geo, geometry='geometry', crs='EPSG:4326')
gage_gdf  = gage_df.merge(summary_df[['gage_id','winner']], on='gage_id').merge(gdf_com, on='COMID')
gage_gdf  = gpd.GeoDataFrame(gage_gdf, geometry='geometry', crs='EPSG:4326').drop_duplicates(subset='gage_id')
states    = gpd.read_file(states_shapefile).to_crs('EPSG:4326')
minx,miny,maxx,maxy = gage_gdf.total_bounds
buffer   = 2
bbox     = box(minx-buffer, miny-buffer, maxx+buffer, maxy+buffer)
states_clipped = gpd.overlay(states, gpd.GeoDataFrame(geometry=[bbox], crs='EPSG:4326'), how='intersection')

# === Plotting layout ===
fig = plt.figure(figsize=(16,10))
gs = gridspec.GridSpec(
    3,3,
    width_ratios=[3,1,1],
    height_ratios=[1,1,0.5],
    wspace=0.3, hspace=0.5
)

# Map on left full height

ax_map = fig.add_subplot(gs[:,0])
states_clipped.plot(ax=ax_map, color='#f7f7f7', edgecolor='black', lw=0.8)
for mdl, grp in gage_gdf.groupby('winner'):
    grp.plot(
        ax=ax_map, marker='o', linestyle='None',
        color=colors[mdl], markersize=60,
        edgecolor='black', linewidth=0.8
    )
ax_map.set_title(
    "Best Performing Model at Each Gage (Validation KGE)",
    fontsize=16
)
ax_map.set_xlabel("Longitude")
ax_map.set_ylabel("Latitude")
ax_map.set_xlim(minx-buffer, maxx+buffer)
ax_map.set_ylim(miny-buffer, maxy+buffer)

if annotate:
    ###this code annotates the gage IDs on the map, can get kind of cluttered so replaced with a legend
    # for _, row in gage_gdf.iterrows():
    #     ax_map.annotate(row["gage_id"], xy=(row.geometry.x, row.geometry.y), fontsize=8)
    # Assign index numbers to gages
    gage_gdf = gage_gdf.reset_index(drop=True)
    gage_gdf["label_num"] = gage_gdf.index + 1  # 1-based numbering

    # Plot number labels near dots
    for _, row in gage_gdf.iterrows():
        num = row["label_num"]
        x, y = row.geometry.x, row.geometry.y
        ax_map.annotate(
            str(num),
            xy=(x, y),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=8,
            ha='left',
            va='bottom'
        )

    # Create the key text (number to gage_id)
    key_text = "\n".join([
        f"{int(row.label_num)}. {row.gage_id}" for _, row in gage_gdf.iterrows()
    ])

    # Add the key as a fixed-position text box
    ax_map.text(
        maxx - 0.5, maxy - 0.3,  # adjust as needed
        f"Gage ID Key:\n{key_text}",
        fontsize=8,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.4')
    )



# Legend with border
lg = ax_map.legend(
    handles=[
        Line2D([],[], marker='o', linestyle='None', color=colors[l], label=l,
               markeredgecolor='black', markeredgewidth=0.8)
        for l in labels[:-1]
    ],
    loc='upper left',
    frameon=True,
    edgecolor='black',
    fontsize=12
)
lg.get_frame().set_linewidth(1)

# Inset model counts table
tbl_map = ax_map.table(
    cellText=[[l, counts.get(l,0)] for l in labels[:-1]],
    colLabels=['Model','# Best'],
    cellLoc='center',
    loc='lower left',
    bbox=[0.02, 0.05, 0.32, 0.20]
)
tbl_map.auto_set_font_size(False)
tbl_map.set_fontsize(10)
for cell in tbl_map.get_celld().values():
    cell.set_linewidth(1)

# Helper for percentiles
def pct(vals):
    return np.percentile(vals, [5,25,50,75,95])

# Create bar-plot axes
ax_kge  = fig.add_subplot(gs[0,1])
ax_vol  = fig.add_subplot(gs[0,2])
ax_peak = fig.add_subplot(gs[1,1])
ax_time = fig.add_subplot(gs[1,2])

# 1) Validation KGE Distribution
kge_data = {
    'LGAR - PF':    summary_df['lasam_val'],
    'CFE':          summary_df['cfe_val'],
    'NWM retro':    summary_df['nwm_val'],
    'Best Overall': summary_df[['lasam_val','cfe_val','nwm_val']].max(axis=1)
}
for i,(lbl,vals) in enumerate(kge_data.items()):
    q5,q25,q50,q75,q95 = pct(vals)
    ax_kge.vlines(i, q5, q95, color='k')
    ax_kge.bar(i, q75-q25, bottom=q25, width=0.6, edgecolor='k', alpha=0.6, color=colors[lbl])
    ax_kge.plot(i, q50, 'o', color='k', ms=5)
    ax_kge.hlines([q5,q95], i-0.3, i+0.3, color='black', linewidth=1.0)
    ax_kge.text(i, q95+0.03, f"{q50:.2f}", ha='center')

wrapped_labels = ['LGAR\nPF', 'CFE', 'NWM\nretro', 'Best\nOverall']
ax_kge.margins(y=0.1)
ax_kge.set_xticks(range(len(wrapped_labels)))
ax_kge.set_xticklabels(wrapped_labels, fontsize=9)
ax_kge.set_ylim(-0.5,1.0)
ax_kge.set_title("Validation KGE")

# 2-4 Error plots
error_plots = [
    (ax_vol,  'volume_error_percent',      "Volume Error (abs %)"),
    (ax_peak, 'peak_flow_error_percent',    "Peak Flow Error (abs %)"),
    (ax_time, 'time_to_peak_error_hours',   "Error in Time To Peak (hr)")
]
for ax, key, title in error_plots:
    data = {lbl: error[lbl][key] for lbl in labels[:-1]}
    data['Best Overall'] = error['Best Overall'][key]
    for i,(lbl,vals) in enumerate(data.items()):
        arr = np.array(vals)
        if lbl == 'Best Overall':
            arr = arr[~np.isnan(arr)]
        q5,q25,q50,q75,q95 = pct(arr)
        ax.vlines(i, q5, q95, color='k')
        ax.bar(i, q75-q25, bottom=q25, width=0.6, edgecolor='k', alpha=0.6, color=colors[lbl])
        ax.plot(i, q50, 'o', color='k', ms=5)
        ax.hlines([q5,q95], i-0.3, i+0.3, color='black', linewidth=1.0)
        ax.text(i, q95 + (1 if 'percent' in key else 0.2), f"{q50:.1f}", ha='center')
    wrapped_labels = ['LGAR\nPF', 'CFE', 'NWM\nretro', 'Best\nOverall\n(KGE\nbased)']
    ax.margins(y=0.1)
    ax.set_xticks(range(len(wrapped_labels)))
    ax.set_xticklabels(wrapped_labels, fontsize=9)
    ax.set_title(title)

# # 5) Summary KGE table under bar grid
# ax_tab = fig.add_subplot(gs[2,1:3])
# ax_tab.axis('off')
# combo = [
#     ["All models",        summary_df[['lasam_val','cfe_val','nwm_val']].max(axis=1).median()],
#     ["Without LGAR - PF", summary_df[['cfe_val','nwm_val']].max(axis=1).median()],
#     ["Without CFE",       summary_df[['lasam_val','nwm_val']].max(axis=1).median()],
#     ["Without NWM retro", summary_df[['lasam_val','cfe_val']].max(axis=1).median()]
# ]

# tbl = ax_tab.table(
#     cellText=[[r[0], f"{r[1]:.3f}"] for r in combo],
#     colLabels=['Model Combo','Median Best KGE'],
#     cellLoc='center',
#     loc='center',
#     bbox=[0.15, 0, 0.7, 1.0]    # narrower (70% of the width)
# )
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(10)           # slightly smaller text
# for cell in tbl.get_celld().values():
#     cell.set_linewidth(1)


# 5) Summary table with all four medians
ax_tab = fig.add_subplot(gs[2,1:3])
ax_tab.axis('off')

# define your four combos and which models they include
combos = {
    'All models':        ['LGAR - PF','CFE','NWM retro'],
    'Without LGAR - PF': ['CFE','NWM retro'],
    'Without CFE':       ['LGAR - PF','NWM retro'],
    'Without NWM retro': ['LGAR - PF','CFE']
}

# prepare the rows: for each combo, compute
#  median of best KGE (i.e. max KGE across included models)
#  median of best volume error (i.e. min abs error across included models)
#  median of best peak flow error
#  median of best time to peak error
cellText = []
for combo_name, models_included in combos.items():
    # best KGE per site = max over the included columns
    cols = {
        'LGAR - PF': 'lasam_val',
        'CFE':       'cfe_val',
        'NWM retro': 'nwm_val'
    }
    kge_series = summary_df[[cols[m] for m in models_included]].max(axis=1)
    med_kge    = kge_series.median()

    # for error metrics: pick the minimum absolute error
    vols  = np.vstack([error[m]['volume_error_percent']      for m in models_included]).T
    peaks = np.vstack([error[m]['peak_flow_error_percent']    for m in models_included]).T
    times = np.vstack([error[m]['time_to_peak_error_hours']   for m in models_included]).T

    med_vol  = np.nanmedian(np.nanmin(vols,  axis=1))
    med_pk   = np.nanmedian(np.nanmin(peaks, axis=1))
    med_time = np.nanmedian(np.nanmin(times, axis=1))

    cellText.append([
        combo_name,
        f"{med_kge:.3f}",
        f"{med_vol:.1f}",
        f"{med_pk:.1f}",
        f"{med_time:.1f}"
    ])

# wrapped, two line column headers
colLabels = [
    "Model\nCombo",
    "Validation\nKGE",
    "Volume Error\n% (abs)",
    "Peak Flow\nError % (abs)",
    "Time To\nPeak (hr)"
]

tbl = ax_tab.table(
    cellText  = cellText,
    colLabels = colLabels,
    cellLoc   = 'center',
    loc       = 'center',
    colWidths = [0.30,0.21,0.21,0.21,0.21]
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for cell in tbl.get_celld().values():
    cell.set_linewidth(1)

# make only the header row taller
header_height = 0.3
ncols = len(colLabels)
for col in range(ncols):
    tbl[(0, col)].set_height(header_height)


tbl.scale(1.0,1.3)

# title
ax_tab.set_title("Median Best Error Metrics", pad=16)


plt.tight_layout()
plt.savefig("best_model_map.png", dpi=300)
plt.show()

print(" Saved updated figure with map, tables, and borders.")





if plot_hydrographs:
    print("\n Generating hydrograph plots with LASAM, CFE, NWM, and Observed...")

    from datetime import datetime

    output_dir = os.path.join(project_root, "model_assessment", "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)

    def parse_nwm_time(s):
        return datetime.strptime(s, "%Y%m%d%H%M")

    kge_lookup = summary_df.set_index("gage_id").to_dict(orient="index")

    for gage_id in summary_df["gage_id"]:
        try:
            obs_path = os.path.join(obs_dir, f"{gage_id}.csv")
            lasam_path = os.path.join(lasam_log_dir.replace("logging", "postproc"), f"{gage_id}_best.csv")
            cfe_path = os.path.join(cfe_log_dir.replace("logging", "postproc"), f"{gage_id}_best.csv")
            comid = gage_to_comid.get(gage_id)

            obs_df = pd.read_csv(obs_path, parse_dates=["value_time"]).set_index("value_time")["flow_m3_per_s"].rename("Observed")
            lasam_df = pd.read_csv(lasam_path, parse_dates=["current_time"]).set_index("current_time")["flow"].rename("LASAM")
            cfe_df = pd.read_csv(cfe_path, parse_dates=["current_time"]).set_index("current_time")["flow"].rename("CFE")

            nwm_df = None
            if comid:
                nwm_path = os.path.join(retro_q_dir, f"{comid}.csv")
                if os.path.exists(nwm_path):
                    nwm_raw = pd.read_csv(nwm_path, converters={"time": parse_nwm_time})
                    nwm_df = nwm_raw.set_index("time")["streamflow"].rename("NWM")

            # Combine and trim
            dfs = [obs_df]
            if nwm_df is not None:
                dfs.append(nwm_df)
            dfs += [cfe_df, lasam_df]

            plot_start = pd.Timestamp("2012-10-01")
            plot_end   = pd.Timestamp("2013-04-30 23:00:00")
            combined = pd.concat(dfs, axis=1).sort_index().loc[plot_start:plot_end].dropna(how="all")

            plt.figure(figsize=(12, 4))
            ax = plt.gca()

            if "NWM" in combined.columns:
                nwm_val = kge_lookup[gage_id].get("nwm_val", None)
                label = f"NWM retrospective (KGE={nwm_val:.2f})" if nwm_val is not None else "NWM retrospective"
                combined["NWM"].plot(ax=ax, color="gold", lw=1, label=label)

            if "CFE" in combined.columns:
                cfe_val = kge_lookup[gage_id].get("cfe_val", None)
                label = f"CFE (KGE={cfe_val:.2f})" if cfe_val is not None else "CFE"
                combined["CFE"].plot(ax=ax, color="blue", lw=1, label=label)

            if "LASAM" in combined.columns:
                lasam_val = kge_lookup[gage_id].get("lasam_val", None)
                label = f"LGAR with simple PF (KGE={lasam_val:.2f})" if lasam_val is not None else "LGAR with simple PF"
                combined["LASAM"].plot(ax=ax, color="green", lw=1, label=label)

            if "Observed" in combined.columns:
                combined["Observed"].plot(ax=ax, color="black", lw=1.2, label="Observed")

            ax.set_title(f"Gage {gage_id}: Observed vs Model Simulations")
            ax.set_ylabel("Flow (m^3/s)")
            ax.set_xlabel("Date")
            ax.grid(True, linestyle="--", linewidth=0.5)

            # Force legend order
            handles, labels = ax.get_legend_handles_labels()
            order = ["NWM", "CFE", "LGAR", "Observed"]
            ordered = sorted(zip(handles, labels), key=lambda x: order.index(next((o for o in order if o in x[1]), x[1])))
            if ordered:
                h, l = zip(*ordered)
                ax.legend(h, l, loc="upper right")

            plt.tight_layout()
            outpath = os.path.join(output_dir, f"{gage_id}_hydrograph.png")
            plt.savefig(outpath, dpi=450)
            plt.close()
            print(f" {gage_id} to {outpath}")

        except Exception as e:
            print(f" Failed to plot {gage_id}: {e}")



