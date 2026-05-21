###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | May 2025]
# updates some NOM parameters in the event that NOM is part of a formulation that is being calibrated

import json

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
