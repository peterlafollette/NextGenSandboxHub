###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# used in compare_and_map to give abs percent error in volume, abs percent error in peak discharge, and error in time to peak as additional error metrics after calibration based on KGE

import numpy as np
import pandas as pd
from hydroeval import kge
# import matplotlib.pyplot as plt

try:
    import xarray as xr
    import neuralhydrology.evaluation.metrics as nhm
    _NH_OK = True
except Exception:
    xr = None
    nhm = None
    _NH_OK = False

def _compute_nh_mappe_fraction(sim: pd.Series, obs: pd.Series) -> float:
    """
    NeuralHydrology MAPPE (mean absolute percentage peak error).
    Returns a fraction clipped to [0, 1], where:
      0.0 = perfect peak reproduction
      1.0 = >=100% peak error (treated as maximally bad)
    """
    if not _NH_OK:
        return np.nan

    sim, obs = sim.align(obs, join="inner")
    m = sim.notna() & obs.notna()
    sim = sim[m]
    obs = obs[m]
    if len(sim) == 0:
        return np.nan

    if not isinstance(sim.index, pd.DatetimeIndex):
        sim = sim.copy()
        sim.index = pd.to_datetime(sim.index)
    if not isinstance(obs.index, pd.DatetimeIndex):
        obs = obs.copy()
        obs.index = pd.to_datetime(obs.index)

    t = sim.index.values
    obs_xr = xr.DataArray(obs.values, coords={"time": t}, dims=("time",))
    sim_xr = xr.DataArray(sim.values, coords={"time": t}, dims=("time",))

    try:
        val = float(nhm.mean_absolute_percentage_peak_error(obs_xr, sim_xr))
    except Exception:
        return np.nan

    return float(np.clip(val / 100, 0.0, 1.0))

def _compute_top_flow_mape_fraction(sim: pd.Series,
                                   obs: pd.Series,
                                   top_frac: float = 0.10,
                                   denom_floor: float = 1e-6) -> float:
    """
    Mean Absolute Percent Error over the top 'top_frac' of observed flows.
    Returns a fraction (0..1) clipped, where 1.0 means >=100% average percent error.

    This avoids tiny-peak issues by only evaluating high-flow timestamps.
    """
    sim, obs = sim.align(obs, join="inner")
    m = sim.notna() & obs.notna()
    sim = sim[m]
    obs = obs[m]
    if len(sim) == 0:
        return np.nan

    # Threshold based on observed distribution (top 10% => 90th percentile)
    q = float(obs.quantile(1.0 - top_frac))
    hi = obs >= q
    if hi.sum() == 0:
        return np.nan

    # Percent error using a denominator floor for extra safety
    denom = np.maximum(obs[hi].to_numpy(dtype=float), denom_floor)
    pe = np.abs(sim[hi].to_numpy(dtype=float) - obs[hi].to_numpy(dtype=float)) / denom  # fraction
    return float(np.clip(np.mean(pe), 0.0, 1.0))


def compute_metrics(sim, obs, event_threshold=1e-2, start_time=None, end_time=None, peak_search_window_hours=10):
    """
    Compute multiple metrics between simulated and observed streamflow,
    optionally restricting to a specific time window.

    Args:
        sim (pd.Series): Simulated streamflow.
        obs (pd.Series): Observed streamflow.
        event_threshold (float): Threshold for defining event flows.
        start_time (str or pd.Timestamp): Start of window.
        end_time (str or pd.Timestamp): End of window.
        peak_search_window_hours (int): Window (in hours) around observed peak for peak-based metrics.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    sim, obs = sim.align(obs, join='inner')
    valid_mask = sim.notna() & obs.notna()
    sim = sim[valid_mask]
    obs = obs[valid_mask]

    if len(sim) == 0 or len(obs) == 0:
        return {
            "kge": -10.0,
            "log_kge": -10.0,
            "volume_error_percent": 999.0,
            "time_to_peak_error_hours": np.nan,
            "peak_flow_error_percent": 999.0,
            "event_kge": -10.0,
            "event_hours": 0,
            "total_hours": 0,
            "mappe": 1.0,
            "dist_to_ideal": np.nan,
            "pareto_kge_mappe": 0.0
        }

    if start_time is not None and end_time is not None:
        sim = sim[start_time:end_time]
        obs = obs[start_time:end_time]

    # Ensure datetime index
    if not isinstance(sim.index, pd.DatetimeIndex):
        sim.index = pd.to_datetime(sim.index)
    if not isinstance(obs.index, pd.DatetimeIndex):
        obs.index = pd.to_datetime(obs.index)

    # === KGE ===
    kge_cal = kge(sim.values, obs.values)[0][0]

    # === NH MAPPE (fraction, clipped <= 1) ===
    # mappe = _compute_nh_mappe_fraction(sim, obs)
    
    # High-flow (top 10% by observed) percent error (fraction)
    mappe = _compute_top_flow_mape_fraction(sim, obs, top_frac=0.10, denom_floor=1e-6)

    # === Distance-to-ideal composite ===
    # ideal point is (KGE=1, MAPPE=0)
    # kge_clip = float(np.clip(kge_cal, -1.0, 1.0))
    kge_eff = float(kge_cal)  # no clipping
    mappe_eff = float(np.clip(mappe, 0.0, 1.0))

    dist_to_ideal = np.sqrt((1.0 - kge_eff)**2 + (mappe_eff)**2)
    pareto_kge_mappe = 1.0 / (1.0 + dist_to_ideal)   # bounded, smooth, no exp()


    # === log-KGE ===
    sim_log = np.log10(np.clip(sim.values, 1e-10, None))
    obs_log = np.log10(np.clip(obs.values, 1e-10, None))
    kge_cal_log = kge(sim_log, obs_log)[0][0]

    # === Volume Error (%) ===
    volume_error_percent = (
        100 * (sim.sum() - obs.sum()) / obs.sum()
        if obs.sum() != 0 else np.nan
    )

    # === Peak Error Metrics in Window ===
    try:
        peak_time_obs = obs.idxmax()
        window_start = peak_time_obs - pd.Timedelta(hours=peak_search_window_hours)
        window_end = peak_time_obs + pd.Timedelta(hours=peak_search_window_hours)

        sim_window = sim[window_start:window_end]

        if not sim_window.empty:
            peak_time_sim = sim_window.idxmax()
            peak_flow_sim = sim_window.max()
        else:
            peak_time_sim = np.nan
            peak_flow_sim = np.nan

        peak_flow_obs = obs.loc[peak_time_obs]

        peak_flow_error_percent = (
            100 * (peak_flow_sim - peak_flow_obs) / peak_flow_obs
            if peak_flow_obs != 0 else np.nan
        )

        time_to_peak_error_hours = (
            abs((peak_time_sim - peak_time_obs).total_seconds()) / 3600.0
            if pd.notna(peak_time_sim) else np.nan
        )

    except Exception:
        peak_flow_error_percent = np.nan
        time_to_peak_error_hours = np.nan

    # === Event-based KGE ===
    try:
        event_mask = (sim > event_threshold) | (obs > event_threshold)
        sim_event = sim[event_mask]
        obs_event = obs[event_mask]

        if len(sim_event) > 0:
            kge_event = kge(sim_event.values, obs_event.values)[0][0]
        else:
            kge_event = -10.0

        event_hours = event_mask.sum()
        total_hours = len(sim)
    except:
        kge_event = -10.0
        event_hours = 0
        total_hours = len(sim)

    return {
        "kge": kge_cal,
        "log_kge": kge_cal_log,
        "volume_error_percent": volume_error_percent,
        "peak_flow_error_percent": peak_flow_error_percent,
        "time_to_peak_error_hours": time_to_peak_error_hours,
        "event_kge": kge_event,
        "event_hours": event_hours,
        "total_hours": total_hours,
        "mappe": mappe_eff,
        "dist_to_ideal": dist_to_ideal,
        "pareto_kge_mappe": pareto_kge_mappe
    }


