###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | May 2025]
# updates some NOM parameters in the event that NOM is part of a formulation that is being calibrated

import glob
import json
import os
import tempfile
from datetime import datetime

NOM_PARAM_ALIASES = {
    "CWP": ("CWP", "CWPVT"),
    "CWPVT": ("CWPVT", "CWP"),
}


def get_nom_param_aliases(param: str):
    return NOM_PARAM_ALIASES.get(param, (param,))


def canonical_nom_param(table_param: str, requested_params):
    for requested in requested_params:
        if table_param in get_nom_param_aliases(requested):
            return requested
    return None


def _matching_update_key(table_param: str, updated_params: dict):
    if table_param in updated_params:
        return table_param
    return canonical_nom_param(table_param, updated_params.keys())


def _format_noahowp_time(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d%H%M")

    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y%m%d%H%M")
        except ValueError:
            pass

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        return f"{digits}0000"
    if len(digits) >= 12:
        return digits[:12]

    raise ValueError(f"Cannot format NoahOWP time from value: {value!r}")


def _is_noahowp_params(params: dict) -> bool:
    model_name = str(params.get("model_type_name", "")).upper()
    init_config = str(params.get("init_config", ""))
    return (
        "NOAHOWP" in model_name
        or "NOAH-OWP" in model_name
        or "NOAH_OWP" in model_name
        or "/noahowp/" in init_config
    )


def _collect_noahowp_init_configs(realization: dict):
    init_configs = set()
    found_noahowp = False

    def _visit(node):
        nonlocal found_noahowp
        if isinstance(node, dict):
            params = node.get("params")
            if isinstance(params, dict) and _is_noahowp_params(params):
                found_noahowp = True
                init_config = params.get("init_config")
                if isinstance(init_config, str) and init_config:
                    init_configs.add(init_config)

            for value in node.values():
                _visit(value)

        elif isinstance(node, list):
            for value in node:
                _visit(value)

    _visit(realization)
    return found_noahowp, sorted(init_configs)


def _expand_noahowp_init_config(init_config: str):
    path = os.path.expandvars(os.path.expanduser(init_config))
    if "{{id}}" in path:
        return sorted(glob.glob(path.replace("{{id}}", "*")))
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(path))


def update_noahowp_input_time_window(config_path: str, start_time, end_time) -> bool:
    """Update startdate/enddate in one NoahOWP namelist-style input file."""
    start = _format_noahowp_time(start_time)
    end = _format_noahowp_time(end_time)

    with open(config_path, "r") as f:
        lines = f.readlines()

    found_start = False
    found_end = False
    updated_lines = []

    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("startdate"):
            updated_lines.append(f'  startdate      = "{start}"  \n')
            found_start = True
        elif stripped.startswith("enddate"):
            updated_lines.append(f'  enddate      = "{end}"  \n')
            found_end = True
        else:
            updated_lines.append(line)

    if not found_start or not found_end:
        missing = []
        if not found_start:
            missing.append("startdate")
        if not found_end:
            missing.append("enddate")
        raise ValueError(f"Missing NoahOWP timing line(s) in {config_path}: {', '.join(missing)}")

    changed = updated_lines != lines
    if changed:
        config_dir = os.path.dirname(os.path.abspath(config_path)) or "."
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(config_path)}.",
            suffix=".tmp",
            dir=config_dir,
            text=True,
        )
        with os.fdopen(fd, "w") as f:
            f.writelines(updated_lines)
        os.replace(tmp_path, config_path)

    return changed


def update_noahowp_time_window(realization_path: str, start_time, end_time) -> int:
    """
    Update NoahOWP namelist start/end dates referenced by an ngen realization.

    PET-only realizations are a no-op. If a NoahOWP module is present but its
    namelist files cannot be found, fail loudly so calibration cannot proceed
    with stale component dates.
    """
    with open(realization_path, "r") as f:
        realization = json.load(f)

    found_noahowp, init_configs = _collect_noahowp_init_configs(realization)
    if not found_noahowp:
        return 0

    config_files = []
    missing_patterns = []
    for init_config in init_configs:
        matches = _expand_noahowp_init_config(init_config)
        if matches:
            config_files.extend(matches)
        else:
            missing_patterns.append(init_config)

    unique_files = sorted(set(config_files))
    if not unique_files:
        raise FileNotFoundError(
            "No NoahOWP input files found for realization "
            f"{realization_path}. Checked: {missing_patterns or init_configs}"
        )

    for config_file in unique_files:
        update_noahowp_input_time_window(config_file, start_time, end_time)

    return len(unique_files)


def update_noahowp_model_params(realization_path: str, updated_params: dict) -> bool:
    """
    Update NoahOWP model_params in an ngen realization JSON.

    This matches the upstream ngen-cal pathway for NoahOWP parameters such as
    SCAMAX that are BMI-settable but are not present as editable rows in
    MPTABLE.TBL.
    """
    if not updated_params:
        return False

    with open(realization_path, "r") as f:
        realization = json.load(f)

    normalized = {str(k): float(v) for k, v in updated_params.items()}

    def _visit(node):
        changed = False
        if isinstance(node, dict):
            params = node.get("params")
            model_name = str(params.get("model_type_name", "")).upper() if isinstance(params, dict) else ""
            if isinstance(params, dict) and "NOAHOWP" in model_name:
                model_params = params.get("model_params")
                if not isinstance(model_params, dict):
                    model_params = {}
                    params["model_params"] = model_params
                model_params.update(normalized)
                changed = True

            for value in node.values():
                changed = _visit(value) or changed

        elif isinstance(node, list):
            for value in node:
                changed = _visit(value) or changed

        return changed

    changed = _visit(realization)
    if changed:
        with open(realization_path, "w") as f:
            json.dump(realization, f, indent=4)

    return changed


def update_mptable(
    original_file: str,
    output_file: str,
    updated_params: dict,
    verbose: bool = False
):
    """
    Replaces values of specific Noah-MP parameters in MPTABLE.TBL without using regex.
    Preserves the number of values per parameter line.

    Parameters:
        original_file (str): Path to the original MPTABLE.TBL.
        output_file (str): Path to write the updated file.
        updated_params (dict): Mapping from parameter name to replacement value or list of values.
        verbose (bool): Whether to print changes.
    """
    with open(original_file, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if '=' in stripped and not stripped.startswith('!'):
            before_eq, after_eq = stripped.split('=', 1)
            param = before_eq.strip()

            update_key = _matching_update_key(param, updated_params)

            if update_key is not None:
                # Handle inline comment
                comment = ''
                if '!' in after_eq:
                    after_eq, comment = after_eq.split('!', 1)
                    comment = f"!{comment.strip()}"

                # Count number of existing values
                raw_values = [val.strip().rstrip(',') for val in after_eq.split(',') if val.strip()]
                n_values = len(raw_values)

                # Get new values: either use provided list or repeat a single value
                new_values = updated_params[update_key]
                if isinstance(new_values, (int, float)):
                    values_to_use = [new_values] * n_values
                elif isinstance(new_values, list):
                    if len(new_values) == 1:
                        values_to_use = new_values * n_values
                    else:
                        values_to_use = new_values[:n_values]
                        if len(values_to_use) < n_values:
                            values_to_use += [new_values[-1]] * (n_values - len(values_to_use))
                else:
                    raise ValueError(f"Unsupported value type for {param}")

                value_str = ',  '.join(f"{v:.6f}" for v in values_to_use) + ','

                # Reconstruct line
                new_line = f"{param:<12} =   {value_str}"
                if comment:
                    new_line += f"  {comment}"
                updated_lines.append(new_line + '\n')

                if verbose:
                    alias_note = "" if update_key == param else f" from {update_key}"
                    print(f"Line {idx + 1}: Updated {param}{alias_note} ({n_values} values)")
                continue

        updated_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(updated_lines)

    if verbose:
        print("\n MPTABLE.TBL successfully overwritten with the following parameter values:")
        for param, val in updated_params.items():
            if isinstance(val, list):
                print(f"    {param}: {val}")
            else:
                print(f"    {param}: {val:.6f}")
        print(f" Written to: {output_file}\n")



# Example usage
if __name__ == "__main__":
    update_mptable(
        original_file="configs/nom/parameters/MPTABLE.TBL",
        output_file="configs/nom/parameters/MPTABLE.TBL",
        updated_params={
            "MFSNO":    [0.6],
            "RSURF_SNOW": [0.005],
            "CWP":      [0.1],
            "VCMX25":   [65.0],
            "MP":       [2],
            "RSURF_EXP": [3],
        },
        verbose=True
    )
