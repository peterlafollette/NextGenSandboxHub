# metrics.py

import numpy as np
import pandas as pd
from hydroeval import kge
import matplotlib.pyplot as plt

def compute_metrics(sim, obs, event_threshold=1e-2, start_time=None, end_time=None):
    """
    Compute multiple metrics between simulated and observed streamflow,
    optionally restricting to a specific time window.

    Args:
        sim (pd.Series): Simulated streamflow, indexed by datetime.
        obs (pd.Series): Observed streamflow, indexed by datetime.
        event_threshold (float): Minimum flow value (in m^3/s) to define an event.
        start_time (str or pd.Timestamp, optional): Start of analysis window.
        end_time (str or pd.Timestamp, optional): End of analysis window.

    Returns:
        dict: Dictionary containing different evaluation metrics.
    """
    # Align sim and obs
    sim, obs = sim.align(obs, join='inner')

    # Drop timestamps with missing values in either series
    valid_mask = sim.notna() & obs.notna()
    sim = sim[valid_mask]
    obs = obs[valid_mask]

    # If no valid data, return placeholder "bad" metrics
    if len(sim) == 0 or len(obs) == 0:
        return {
            "kge": -10.0,
            "log_kge": -10.0,
            "volume_error_percent": -9999.0,
            "peak_time_error_hours": -9999.0,
            "peak_flow_error_percent": -9999.0,
            "event_kge": -10.0,
            "event_hours": 0,
            "total_hours": 0
        }

    # Restrict to the desired time window if provided
    if start_time is not None and end_time is not None:
        sim = sim[start_time:end_time]
        obs = obs[start_time:end_time]

    # --- Basic KGE
    kge_cal = kge(sim.values, obs.values)[0][0]

    # --- Log-transformed KGE
    sim_log = np.log10(np.clip(sim.values, 1e-10, None))
    obs_log = np.log10(np.clip(obs.values, 1e-10, None))
    kge_cal_log = kge(sim_log, obs_log)[0][0]

    # --- Event-based KGE
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
    except Exception:
        kge_event = -10.0
        event_hours = np.nan
        total_hours = np.nan

    # === Return metrics (with placeholders for deprecated ones)
    return {
        "kge": kge_cal,
        "log_kge": kge_cal_log,
        "volume_error_percent": -9999.0,
        "peak_time_error_hours": -9999.0,
        "peak_flow_error_percent": -9999.0,
        "event_kge": kge_event,
        "event_hours": event_hours,
        "total_hours": total_hours
    }
