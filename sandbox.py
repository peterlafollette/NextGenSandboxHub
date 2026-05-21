############################################################################################
# Author  : Ahmad Jan Khattak
# Contact : ahmad.jan.khattak@noaa.gov
# Date    : July 16, 2024
############################################################################################
###############################################################
# edits from: Peter La Follette [plafollette@lynker.com | May 2025], to allow for tiled formulations
# edits (Aug 2025): concurrent particle scaffolding (-conf) and targeting a particle (-run)

import os, sys
import subprocess
import yaml
import argparse
from pathlib import Path
import pandas as pd
import shutil
import glob
import platform

path = Path(sys.argv[0]).resolve()
sandbox_dir = path.parent

from src.python import forcing, driver, runner

def _configured_gage_list_path():
    gage_file = os.environ.get("BASIN_CSV")
    if gage_file:
        return Path(os.path.expanduser(os.path.expandvars(gage_file)))
    return sandbox_dir / "basin_IDs" / "basin_IDs.csv"

def _update_nom_namelist_paramdir(namelist_path: Path, new_param_dir: Path):
    """Update `parameter_dir = "..."` in a NOM Fortran namelist file."""
    if not namelist_path.is_file():
        return
    lines = []
    with namelist_path.open("r") as f:
        for line in f:
            if ("parameter_dir" in line) and ("=" in line) and (not line.strip().startswith("!")):
                quote = '"' if '"' in line else "'"
                # preserve any trailing inline comment
                before, after = line.split("=", 1)
                comment = ""
                if "!" in after:
                    after, comment = after.split("!", 1)
                    comment = "  !" + comment.strip()
                line = f'{before}= {quote}{str(new_param_dir)}{quote}{comment}\n'
            lines.append(line)
    with namelist_path.open("w") as f:
        f.writelines(lines)

def _seed_particle_pet_configs(gage_path: Path, particle_root: Path):
    """
    Copy gage-level PET configs into the particle workspace.
    Source:  <gage>/configs/pet/
    Target:  <gage>/particles/pX/configs/pet/
    """
    base_pet_dir = gage_path / "configs" / "pet"
    if not base_pet_dir.is_dir():
        return  # gage does not use PET

    dst_pet_dir = particle_root / "configs" / "pet"
    dst_pet_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in sorted(base_pet_dir.glob("*")):
        if src.is_file():
            try:
                shutil.copy2(src, dst_pet_dir / src.name)
                copied += 1
            except Exception as e:
                print(f"[conf] Warning copying PET config {src.name}: {e}")
    if copied:
        print(f"[conf] Seeded {copied} PET file(s) into {dst_pet_dir}")

def _retarget_lasam_file_fields(cfg_path: Path, particle_cfg_dir: Path):
    """
    In a single lasam_config_cat* file, find any '<key>=<path>' entries for keys that
    look like file pointers (e.g., '*_file'), copy those files into the particle's
    LASAM config dir, and rewrite the line to point at the particle-local absolute path.

    This is conservative (copy-only) and safe for concurrent particles.
    """
    if not cfg_path.is_file():
        return

    lines = cfg_path.read_text().splitlines()
    new_lines = []
    changed = False

    for line in lines:
        stripped = line.strip()
        # skip blank/comment lines
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue

        if "=" not in stripped:
            new_lines.append(line)
            continue

        key, raw_val = stripped.split("=", 1)
        key = key.strip()
        val = raw_val.strip()

        # treat any key ending with '_file' as a file pointer (e.g., 'soil_params_file')
        if key.endswith("_file"):
            src = Path(val)
            if not src.is_absolute():
                src = (cfg_path.parent / src).resolve()

            try:
                particle_cfg_dir.mkdir(parents=True, exist_ok=True)
                dst = particle_cfg_dir / src.name
                if src.is_file() and not dst.exists():
                    shutil.copy2(src, dst)
                # rewrite to particle-local absolute path
                new_lines.append(f"{key}={str(dst)}")
                changed = True
            except Exception as e:
                print(f"[conf] Warning: could not seed particle LASAM file for {cfg_path.name}: {e}")
                new_lines.append(line)  # fallback
        else:
            new_lines.append(line)

    if changed:
        cfg_path.write_text("\n".join(new_lines) + "\n")


def _seed_particle_lasam_configs(gage_path: Path, particle_root: Path):
    """
    Copy gage-level LASAM configs into particle workspace and retarget file fields.
    Source:  <gage>/configs/lasam/
    Target:  <gage>/particles/pX/configs/lasam/
    """
    base_lasam_dir = gage_path / "configs" / "lasam"
    if not base_lasam_dir.is_dir():
        return  # gage does not use LASAM

    dst_lasam_dir = particle_root / "configs" / "lasam"
    dst_lasam_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in sorted(base_lasam_dir.glob("lasam_config_cat*")):
        try:
            shutil.copy2(src, dst_lasam_dir / src.name)
            copied += 1
        except Exception as e:
            print(f"[conf] Warning copying LASAM config {src.name}: {e}")

    if copied:
        for cfg_copy in sorted(dst_lasam_dir.glob("lasam_config_cat*")):
            _retarget_lasam_file_fields(cfg_copy, dst_lasam_dir)
        print(f"[conf] Seeded {copied} LASAM config(s) into {dst_lasam_dir}")


def _seed_particle_casam_configs(gage_path: Path, particle_root: Path):
    """
    Copy gage-level CASAM configs into particle workspace and retarget file fields.
    Source:  <gage>/configs/casam/
    Target:  <gage>/particles/pX/configs/casam/
    """
    base_casam_dir = gage_path / "configs" / "casam"
    if not base_casam_dir.is_dir():
        return  # gage does not use CASAM

    dst_casam_dir = particle_root / "configs" / "casam"
    dst_casam_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    patterns = ("casam_cfg_cat*", "casam_config_cat*")
    seen = set()
    for pattern in patterns:
        for src in sorted(base_casam_dir.glob(pattern)):
            if src in seen:
                continue
            seen.add(src)
            try:
                shutil.copy2(src, dst_casam_dir / src.name)
                copied += 1
            except Exception as e:
                print(f"[conf] Warning copying CASAM config {src.name}: {e}")

    if copied:
        for cfg_copy in sorted(dst_casam_dir.glob("casam_*cat*")):
            _retarget_lasam_file_fields(cfg_copy, dst_casam_dir)
        print(f"[conf] Seeded {copied} CASAM config(s) into {dst_casam_dir}")


def CheckSandbox_VENV():
    VENV_SANDBOX = Path.home() / ".venv_sandbox_py3.11"

    # Check if the virtual environment exists
    if not VENV_SANDBOX.exists():
        print(f"Error: NextGen virtual environment {VENV_SANDBOX} not found under home directory...")
        sys.exit(1)

    # Check if the script is running inside that required environment
    VENV_ACTIVE = Path(sys.prefix)
    if VENV_ACTIVE.resolve() != VENV_SANDBOX.resolve():
        print(f"Warning: sandbox.py is not running in the expected Python virtual environment.")
        print(f"Expected: {VENV_SANDBOX}")
        print(f"Active:   {VENV_ACTIVE}")
        sys.exit(1)


formulations_supported = [
    "NOM,CFE",
    "PET,CFE",
    "NOM,LASAM",
    "PET,LASAM",
    "NOM,CASAM",
    "PET,CASAM",
    "NOM,CFE,PET",
    "NOM,CFE,SMP,SFT",
    "NOM,LASAM,SMP,SFT",
    "NOM,CASAM,SMP,SFT",
    "NOM,TOPMODEL",
    "BASELINE,CFE",
    "BASELINE,LAS"
]

def Sandbox(sandbox_config, calib_config):

    if (args.subset):
        print ("Generating geopackages...")
        subset_basin = f"Rscript {sandbox_dir}/src/R/main.R {sandbox_config}"
        status = subprocess.call(subset_basin, shell=True)

        if (status):
            sys.exit("Failed during generating geopackge(s) step...")
        else:
            print ("DONE \u2713")

    if (args.forc):
        print ("Generating forcing data...")
        process_forcing = forcing.ForcingProcessor(sandbox_config)
        status          = process_forcing.download_forcing()

        if (status):
            sys.exit("Failed during generating geopackge(s) step...")
        else:
            print ("DONE \u2713")

    if (args.conf):
        print ("Generating config files...")
        _driver = driver.Driver(sandbox_config, formulations_supported, gage_id=args.gage_id)
        status  = _driver.run()

        if (status):
            sys.exit("Failed during generating config files step...")
        else:
            print ("DONE \u2713")

            # Disable Spotlight indexing in outputs/div and troute if they exist or might be used
            def disable_spotlight_indexing(path_str):
                path = Path(path_str)
                if platform.system() == "Darwin":
                    path.mkdir(parents=True, exist_ok=True)
                    (path / ".metadata_never_index").touch()

            # Parse output_dir from the sandbox config file
            with open(sandbox_config, "r") as f:
                sandbox_dict = yaml.safe_load(f)
            output_base = Path(os.path.expandvars(sandbox_dict["output_dir"]))

            print("output_base")
            print(output_base)

            # Create postproc directory at same level as output_dir
            postproc_dir = output_base.parent / "postproc"
            postproc_dir.mkdir(parents=True, exist_ok=True)
            disable_spotlight_indexing(postproc_dir)

            # Apply to all gage output dirs created
            gage_div_paths = glob.glob(str(output_base / "*" / "outputs" / "div"))
            for div_path in gage_div_paths:
                disable_spotlight_indexing(div_path)

            for gage_path in output_base.glob("*"):
                if not gage_path.is_dir():
                    continue  # skip files like .DS_Store
                div_path = gage_path / "outputs" / "div"
                troute_path = gage_path / "troute"

                disable_spotlight_indexing(div_path)

                if not troute_path.exists():
                    try:
                        troute_path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        print(f"Warning: Could not create troute dir: {troute_path} - {e}")
                disable_spotlight_indexing(troute_path)

            # --- Filter gages to those listed in the configured gage CSV ---
            gage_list_path = _configured_gage_list_path()
            with open(sandbox_config, "r") as f:
                sandbox_dict = yaml.safe_load(f)
            output_base = Path(os.path.expandvars(sandbox_dict["output_dir"]))

            selected = None
            if getattr(args, "gage_id", None):
                selected = {str(args.gage_id).strip()}
            elif gage_list_path.exists():
                try:
                    ids_df = pd.read_csv(gage_list_path, dtype=str)
                    selected = set(ids_df["gage_id"].astype(str).str.strip())
                except Exception as e:
                    print(f"Warning: could not read {gage_list_path}: {e}")

            if selected:
                kept = 0
                for gage_path in output_base.glob("*"):
                    if not gage_path.is_dir():
                        continue
                    gid = gage_path.name  # assumes out/<gage_id>/...
                    if gid not in selected:
                        shutil.rmtree(gage_path, ignore_errors=True)
                    else:
                        kept += 1
                print(f"[conf] Kept {kept} gage(s) from {gage_list_path}; removed others.")
            else:
                print(f"Warning: {gage_list_path} missing or unreadable; no gage filtering applied.")


            ###
            # ===== Per-particle workspace scaffolding (optional) =====
            if getattr(args, "concurrent_particles", False) and getattr(args, "num_particles", 1) >= 1:
                num_p = int(args.num_particles)
                print(f"[conf] Concurrent particle mode enabled. Scaffolding {num_p} particle workspaces per gage...")

                def particle_root(base_gage_dir: Path, pid: int) -> Path:
                    return base_gage_dir / "particles" / f"p{pid}"

                # For every gage produced in output_base
                for gage_path in output_base.glob("*"):
                    if not gage_path.is_dir():
                        continue

                    # Likely locations after driver.run()
                    cfg_dir      = gage_path / "configs"
                    cfg_cfe_dir  = cfg_dir / "cfe"
                    cfg_troute   = cfg_dir / "troute_config.yaml"
                    json_dir     = gage_path / "json"

                    # ADD THESE:
                    cfg_pet_dir = cfg_dir / "pet"
                    nom_cfg_dir = cfg_dir / "noahowp"
                    nom_tbl     = nom_cfg_dir / "parameters" / "MPTABLE.TBL"


                    # Probe existing realization JSON (first *.json)
                    json_src = None
                    if json_dir.exists():
                        json_files = list(json_dir.glob("*.json"))
                        if json_files:
                            json_src = json_files[0]

                    ###
                    for pid in range(num_p):
                        prow = particle_root(gage_path, pid)
                        (prow / "configs").mkdir(parents=True, exist_ok=True)
                        (prow / "json").mkdir(parents=True, exist_ok=True)
                        (prow / "outputs" / "div").mkdir(parents=True, exist_ok=True)
                        (prow / "outputs" / "div_weighted").mkdir(parents=True, exist_ok=True)
                        (prow / "troute").mkdir(parents=True, exist_ok=True)
                        (prow / "postproc").mkdir(parents=True, exist_ok=True)

                        for p in [prow / "outputs" / "div",
                                prow / "outputs" / "div_weighted",
                                prow / "troute",
                                prow / "postproc"]:
                            disable_spotlight_indexing(p)


                        # CFE (unchanged)
                        if cfg_cfe_dir.exists():
                            (prow / "configs" / "cfe").mkdir(parents=True, exist_ok=True)
                            for src in sorted(cfg_cfe_dir.glob("cfe_config_cat*")):
                                try:
                                    shutil.copy2(src, (prow / "configs" / "cfe" / src.name))
                                except Exception as e:
                                    print(f"[conf][p{pid}] Warning copying {src}: {e}")

                        # PET
                        try:
                            _seed_particle_pet_configs(gage_path, prow)
                        except Exception as e:
                            print(f"[conf][p{pid}] Warning seeding PET configs: {e}")

                        try:
                            _seed_particle_lasam_configs(gage_path, prow)
                        except Exception as e:
                            print(f"[conf][p{pid}] Warning seeding LASAM configs: {e}")

                        try:
                            _seed_particle_casam_configs(gage_path, prow)
                        except Exception as e:
                            print(f"[conf][p{pid}] Warning seeding CASAM configs: {e}")

                        # troute (unchanged)
                        if cfg_troute.exists():
                            try:
                                shutil.copy2(cfg_troute, (prow / "configs" / "troute_config.yaml"))
                            except Exception as e:
                                print(f"[conf][p{pid}] Warning copying troute_config.yaml: {e}")

                        # NOM: ONLY if gage actually has NOM in base
                        if nom_cfg_dir.exists():
                            # Copy MPTABLE if present
                            if nom_tbl.exists():
                                dst_nom = prow / "configs" / "noahowp" / "parameters"
                                dst_nom.mkdir(parents=True, exist_ok=True)
                                try:
                                    shutil.copy2(nom_tbl, dst_nom / "MPTABLE.TBL")
                                except Exception as e:
                                    print(f"[conf][p{pid}] Warning copying MPTABLE.TBL: {e}")

                            # particle NOM locations
                            p_nom_cfg_dir   = prow / "configs" / "noahowp"
                            p_nom_param_dir = p_nom_cfg_dir / "parameters"
                            p_nom_cfg_dir.mkdir(parents=True, exist_ok=True)
                            p_nom_param_dir.mkdir(parents=True, exist_ok=True)

                            # Link/copy GENPARM/SOILPARM if present at base
                            nom_param_dir = nom_cfg_dir / "parameters"
                            for tbl in ("GENPARM.TBL", "SOILPARM.TBL"):
                                src_tbl = nom_param_dir / tbl
                                dst_tbl = p_nom_param_dir / tbl
                                if src_tbl.is_file() and not dst_tbl.exists():
                                    try:
                                        os.symlink(src_tbl, dst_tbl)
                                    except OSError:
                                        try:
                                            shutil.copy2(src_tbl, dst_tbl)
                                        except Exception as e:
                                            print(f"[conf][p{pid}] Warning preparing {tbl}: {e}")

                            # Copy any NOM namelists + retarget parameter_dir
                            nom_namelists = list(nom_cfg_dir.glob("*.input")) + list(nom_cfg_dir.glob("noah*.in*"))
                            for src_nl in nom_namelists:
                                dst_nl = p_nom_cfg_dir / src_nl.name
                                try:
                                    shutil.copy2(src_nl, dst_nl)
                                    _update_nom_namelist_paramdir(dst_nl, p_nom_param_dir)
                                except Exception as e:
                                    print(f"[conf][p{pid}] Warning copying/patching NOM namelist {src_nl.name}: {e}")

                        # realization JSON (unchanged)
                        if json_src and json_src.exists():
                            try:
                                shutil.copy2(json_src, (prow / "json" / json_src.name))
                            except Exception as e:
                                print(f"[conf][p{pid}] Warning copying realization JSON: {e}")


                print("[conf] Particle workspaces ready.")

            # Create a filtered basins_passed_custom.csv file
            passed_basins_csv = output_base / "basins_passed.csv"
            custom_csv = output_base / "basins_passed_custom.csv"
            gage_list_path = _configured_gage_list_path()

            selected_gage_ids = None
            if getattr(args, "gage_id", None):
                selected_gage_ids = {str(args.gage_id).strip()}
            elif gage_list_path.exists():
                try:
                    gage_ids_df = pd.read_csv(gage_list_path, dtype=str)
                    selected_gage_ids = set(gage_ids_df['gage_id'].astype(str).str.strip())
                except Exception as e:
                    print(f"Warning: Failed to read {gage_list_path} - {e}")

            if passed_basins_csv.exists() and selected_gage_ids:
                try:
                    # Read basins_passed.csv
                    basins_df = pd.read_csv(passed_basins_csv, dtype=str)

                    # Determine matching column name in basins_passed.csv
                    possible_id_cols = ["STAID", "gage_id", "gageID", "id"]
                    matching_col = next((col for col in possible_id_cols if col in basins_df.columns), None)

                    if matching_col is None:
                        print(f"Warning: No matching gage ID column found in {passed_basins_csv}")
                    else:
                        filtered_df = basins_df[basins_df[matching_col].astype(str).isin(selected_gage_ids)]
                        filtered_df.to_csv(custom_csv, index=False)
                        print(f"Filtered basins_passed_custom.csv created at {custom_csv} with {len(filtered_df)} basins.")
                except Exception as e:
                    print(f"Warning: Failed to create filtered basins_passed_custom.csv - {e}")
            else:
                if not passed_basins_csv.exists():
                    print(f"Warning: basins_passed.csv not found at {passed_basins_csv}")
                if not selected_gage_ids:
                    print(f"Warning: configured gage CSV not found at {gage_list_path}")

    if (args.run):
        print ("Calling Runner...")

        # Pass particle selection through environment so Runner can opt-in without API breakage
        if getattr(args, "concurrent_particles", False):
            os.environ["NGEN_CONCURRENT_PARTICLES"] = "1"
        if getattr(args, "particle_id", None) is not None:
            os.environ["NGEN_PARTICLE_ID"] = str(args.particle_id)

        _runner = runner.Runner(sandbox_config, calib_config, gage_id=args.gage_id)
        status  = _runner.run()

        if (status):
            sys.exit("Failed during ngen-cal execution...")
        else:
            print ("DONE \u2713")

    print ("**********************************")


if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-subset", action='store_true',    help="Subset basin (generate .gpkg files)")
        parser.add_argument("-forc",   action='store_true',    help="Download forcing data")
        parser.add_argument("-conf",   action='store_true',    help="Generate config files")
        parser.add_argument("-run",    action='store_true',    help="Run NextGen simulations")
        parser.add_argument("-i",      dest="sandbox_infile", type=str, required=False,  help="sandbox config file")
        parser.add_argument("-j",      dest="calib_infile",    type=str, required=False,  help="caliberation config file")
        parser.add_argument("--gage_id", type=str, required=False, help="Run model only for this gage ID")
        # New concurrent-eval flags
        parser.add_argument("--concurrent-particles", action="store_true",
                            help="Enable per-particle workspaces for concurrent PSO evaluation")
        parser.add_argument("--num-particles", type=int, default=1,
                            help="Number of particle workspaces to scaffold at -conf")
        parser.add_argument("--particle-id", type=int, default=None,
                            help="Target particle workspace when running (-run)")
        args = parser.parse_args()
    except SystemExit:
        print("Formulations supported:\n" + "\n".join(formulations_supported))
        sys.exit(0)

    if (args.sandbox_infile):
        if (os.path.exists(args.sandbox_infile)):
            sandbox_config = Path(args.sandbox_infile).resolve()
        else:
            print ("sandbox config file DOES NOT EXIST, provided: ", args.sandbox_infile)
            sys.exit(0)
    else:
        sandbox_config = f"{sandbox_dir}/configs/sandbox_config.yaml"

    if (args.calib_infile):
        if (os.path.exists(args.calib_infile)):
            calib_config = Path(args.calib_infile).resolve()
        else:
            print ("caliberation config file DOES NOT EXIST, provided: ", args.calib_infile)
            sys.exit(0)
    else:
        calib_config = f"{sandbox_dir}/configs/calib_config.yaml"

    if (len(sys.argv) < 2):
        print ("No arguments are provide")
        sys.exit(0)

    # check if expected Python virtual env exists and activated
    CheckSandbox_VENV()

    Sandbox(sandbox_config, calib_config)
































# ############################################################################################
# # Author  : Ahmad Jan Khattak
# # Contact : ahmad.jan.khattak@noaa.gov
# # Date    : July 16, 2024
# ############################################################################################
# ###############################################################
# # edits from: Peter La Follette [plafollette@lynker.com | May 2025], to allow for tiled formulations
# # edits (Aug 2025): concurrent particle scaffolding (-conf) and targeting a particle (-run)

# import os, sys
# import subprocess
# import yaml
# import argparse
# from pathlib import Path
# import pandas as pd
# import shutil
# import glob
# import platform

# path = Path(sys.argv[0]).resolve()
# sandbox_dir = path.parent

# from src.python import forcing, driver, runner

# def _update_nom_namelist_paramdir(namelist_path: Path, new_param_dir: Path):
#     """Update `parameter_dir = "..."` in a NOM Fortran namelist file."""
#     if not namelist_path.is_file():
#         return
#     lines = []
#     with namelist_path.open("r") as f:
#         for line in f:
#             if ("parameter_dir" in line) and ("=" in line) and (not line.strip().startswith("!")):
#                 quote = '"' if '"' in line else "'"
#                 # preserve any trailing inline comment
#                 before, after = line.split("=", 1)
#                 comment = ""
#                 if "!" in after:
#                     after, comment = after.split("!", 1)
#                     comment = "  !" + comment.strip()
#                 line = f'{before}= {quote}{str(new_param_dir)}{quote}{comment}\n'
#             lines.append(line)
#     with namelist_path.open("w") as f:
#         f.writelines(lines)



# def CheckSandbox_VENV():
#     VENV_SANDBOX = Path.home() / ".venv_sandbox_py3.11"

#     # Check if the virtual environment exists
#     if not VENV_SANDBOX.exists():
#         print(f"Error: NextGen virtual environment {VENV_SANDBOX} not found under home directory...")
#         sys.exit(1)

#     # Check if the script is running inside that required environment
#     VENV_ACTIVE = Path(sys.prefix)
#     if VENV_ACTIVE.resolve() != VENV_SANDBOX.resolve():
#         print(f"Warning: sandbox.py is not running in the expected Python virtual environment.")
#         print(f"Expected: {VENV_SANDBOX}")
#         print(f"Active:   {VENV_ACTIVE}")
#         sys.exit(1)


# formulations_supported = [
#     "NOM,CFE",
#     "PET,CFE",
#     "NOM,LASAM",
#     "PET,LASAM",
#     "NOM,CFE,PET",
#     "NOM,CFE,SMP,SFT",
#     "NOM,LASAM,SMP,SFT",
#     "NOM,TOPMODEL",
#     "BASELINE,CFE",
#     "BASELINE,LAS"
# ]

# def Sandbox(sandbox_config, calib_config):

#     if (args.subset):
#         print ("Generating geopackages...")
#         subset_basin = f"Rscript {sandbox_dir}/src/R/main.R {sandbox_config}"
#         status = subprocess.call(subset_basin, shell=True)

#         if (status):
#             sys.exit("Failed during generating geopackge(s) step...")
#         else:
#             print ("DONE \u2713")

#     if (args.forc):
#         print ("Generating forcing data...")
#         process_forcing = forcing.ForcingProcessor(sandbox_config)
#         status          = process_forcing.download_forcing()

#         if (status):
#             sys.exit("Failed during generating geopackge(s) step...")
#         else:
#             print ("DONE \u2713")

#     if (args.conf):
#         print ("Generating config files...")
#         _driver = driver.Driver(sandbox_config, formulations_supported)
#         status  = _driver.run()

#         if (status):
#             sys.exit("Failed during generating config files step...")
#         else:
#             print ("DONE \u2713")

#             # Disable Spotlight indexing in outputs/div and troute if they exist or might be used
#             def disable_spotlight_indexing(path_str):
#                 path = Path(path_str)
#                 if platform.system() == "Darwin":
#                     path.mkdir(parents=True, exist_ok=True)
#                     (path / ".metadata_never_index").touch()

#             # Parse output_dir from the sandbox config file
#             with open(sandbox_config, "r") as f:
#                 sandbox_dict = yaml.safe_load(f)
#             output_base = Path(sandbox_dict["output_dir"])

#             print("output_base")
#             print(output_base)

#             # Create postproc directory at same level as output_dir
#             postproc_dir = output_base.parent / "postproc"
#             postproc_dir.mkdir(parents=True, exist_ok=True)
#             disable_spotlight_indexing(postproc_dir)

#             # Apply to all gage output dirs created
#             gage_div_paths = glob.glob(str(output_base / "*" / "outputs" / "div"))
#             for div_path in gage_div_paths:
#                 disable_spotlight_indexing(div_path)

#             for gage_path in output_base.glob("*"):
#                 if not gage_path.is_dir():
#                     continue  # skip files like .DS_Store
#                 div_path = gage_path / "outputs" / "div"
#                 troute_path = gage_path / "troute"

#                 disable_spotlight_indexing(div_path)

#                 if not troute_path.exists():
#                     try:
#                         troute_path.mkdir(parents=True, exist_ok=True)
#                     except Exception as e:
#                         print(f"Warning: Could not create troute dir: {trroute_path} - {e}")
#                 disable_spotlight_indexing(troute_path)

#             # --- Filter gages to those listed in basin_IDs.csv ---
#             gage_list_path = sandbox_dir / "basin_IDs" / "basin_IDs.csv"
#             with open(sandbox_config, "r") as f:
#                 sandbox_dict = yaml.safe_load(f)
#             output_base = Path(sandbox_dict["output_dir"])

#             selected = None
#             if gage_list_path.exists():
#                 try:
#                     ids_df = pd.read_csv(gage_list_path, dtype=str)
#                     selected = set(ids_df["gage_id"].astype(str).str.strip())
#                 except Exception as e:
#                     print(f"Warning: could not read {gage_list_path}: {e}")

#             if selected:
#                 kept = 0
#                 for gage_path in output_base.glob("*"):
#                     if not gage_path.is_dir():
#                         continue
#                     gid = gage_path.name  # assumes out/<gage_id>/...
#                     if gid not in selected:
#                         shutil.rmtree(gage_path, ignore_errors=True)
#                     else:
#                         kept += 1
#                 print(f"[conf] Kept {kept} gage(s) from basin_IDs.csv; removed others.")
#             else:
#                 print("Warning: basin_IDs.csv missing or unreadable; no gage filtering applied.")


#             ###
#             # ===== Per-particle workspace scaffolding (optional) =====
#             if getattr(args, "concurrent_particles", False) and getattr(args, "num_particles", 1) > 1:
#                 num_p = int(args.num_particles)
#                 print(f"[conf] Concurrent particle mode enabled. Scaffolding {num_p} particle workspaces per gage...")

#                 def particle_root(base_gage_dir: Path, pid: int) -> Path:
#                     return base_gage_dir / "particles" / f"p{pid}"

#                 # For every gage produced in output_base
#                 for gage_path in output_base.glob("*"):
#                     if not gage_path.is_dir():
#                         continue

#                     # Likely locations after driver.run()
#                     cfg_dir      = gage_path / "configs"
#                     cfg_cfe_dir  = cfg_dir / "cfe"
#                     cfg_troute   = cfg_dir / "troute_config.yaml"
#                     nom_tbl      = gage_path / "configs" / "noahowp" / "parameters" / "MPTABLE.TBL"
#                     json_dir     = gage_path / "json"

#                     # Probe existing realization JSON (first *.json)
#                     json_src = None
#                     if json_dir.exists():
#                         json_files = list(json_dir.glob("*.json"))
#                         if json_files:
#                             json_src = json_files[0]

#                     for pid in range(num_p):
#                         prow = particle_root(gage_path, pid)
#                         # Create directory tree
#                         (prow / "configs").mkdir(parents=True, exist_ok=True)
#                         (prow / "json").mkdir(parents=True, exist_ok=True)
#                         (prow / "outputs" / "div").mkdir(parents=True, exist_ok=True)
#                         (prow / "outputs" / "div_weighted").mkdir(parents=True, exist_ok=True)
#                         (prow / "troute").mkdir(parents=True, exist_ok=True)
#                         (prow / "postproc").mkdir(parents=True, exist_ok=True)

#                         # Disable Spotlight indexing (macOS) for heavy I/O dirs
#                         for p in [prow / "outputs" / "div",
#                                   prow / "outputs" / "div_weighted",
#                                   prow / "troute",
#                                   prow / "postproc"]:
#                             disable_spotlight_indexing(p)

#                         # Copy CFE configs (mutable)
#                         if cfg_cfe_dir.exists():
#                             (prow / "configs" / "cfe").mkdir(parents=True, exist_ok=True)
#                             for src in sorted(cfg_cfe_dir.glob("cfe_config_cat*")):
#                                 try:
#                                     shutil.copy2(src, prow / "configs" / "cfe" / src.name)
#                                 except Exception as e:
#                                     print(f"[conf][p{pid}] Warning copying {src}: {e}")

#                         # Copy troute_config.yaml if present (mutable)
#                         if cfg_troute.exists():
#                             try:
#                                 shutil.copy2(cfg_troute, prow / "configs" / "troute_config.yaml")
#                             except Exception as e:
#                                 print(f"[conf][p{pid}] Warning copying troute_config.yaml: {e}")

#                         # Copy NOM table if present (mutable)
#                         if nom_tbl.exists():
#                             dst_nom = prow / "configs" / "noahowp" / "parameters"
#                             dst_nom.mkdir(parents=True, exist_ok=True)
#                             try:
#                                 shutil.copy2(nom_tbl, dst_nom / "MPTABLE.TBL")
#                             except Exception as e:
#                                 print(f"[conf][p{pid}] Warning copying MPTABLE.TBL: {e}")

#                         # --- NOM namelist(s) into particle + retarget parameter_dir ---
#                         # gage-level NOM locations
#                         nom_cfg_dir   = gage_path / "configs" / "noahowp"
#                         nom_param_dir = nom_cfg_dir / "parameters"

#                         # particle NOM locations
#                         p_nom_cfg_dir   = prow / "configs" / "noahowp"
#                         p_nom_param_dir = p_nom_cfg_dir / "parameters"
#                         p_nom_cfg_dir.mkdir(parents=True, exist_ok=True)
#                         p_nom_param_dir.mkdir(parents=True, exist_ok=True)

#                         # 1) Ensure parameter tables exist for the particle:
#                         #    - prefer symlinks to save inodes, fall back to copy.
#                         for tbl in ("GENPARM.TBL", "SOILPARM.TBL"):
#                             src_tbl = nom_param_dir / tbl
#                             dst_tbl = p_nom_param_dir / tbl
#                             if src_tbl.is_file() and not dst_tbl.exists():
#                                 try:
#                                     os.symlink(src_tbl, dst_tbl)
#                                 except OSError:
#                                     try:
#                                         shutil.copy2(src_tbl, dst_tbl)
#                                     except Exception as e:
#                                         print(f"[conf][p{pid}] Warning preparing {tbl}: {e}")

#                         # MPTABLE.TBL is already handled above (copied into p_nom_param_dir)

#                         # 2) Copy any NOM namelist(s) used by the realization into particle dir.
#                         #    These typically look like noahowp_config_cat-*.input, but we’ll be liberal:
#                         nom_namelists = list(nom_cfg_dir.glob("*.input")) + list(nom_cfg_dir.glob("noah*.in*"))
#                         for src_nl in nom_namelists:
#                             dst_nl = p_nom_cfg_dir / src_nl.name
#                             try:
#                                 shutil.copy2(src_nl, dst_nl)
#                                 # 3) Retarget parameter_dir inside the particle namelist
#                                 _update_nom_namelist_paramdir(dst_nl, p_nom_param_dir)
#                             except Exception as e:
#                                 print(f"[conf][p{pid}] Warning copying/patching NOM namelist {src_nl.name}: {e}")


#                         # Copy realization JSON (mutable)
#                         if json_src and json_src.exists():
#                             try:
#                                 shutil.copy2(json_src, (prow / "json" / json_src.name))
#                             except Exception as e:
#                                 print(f"[conf][p{pid}] Warning copying realization JSON: {e}")

#                 print("[conf] Particle workspaces ready.")

#             # Create a filtered basins_passed_custom.csv file
#             passed_basins_csv = output_base / "basins_passed.csv"
#             custom_csv = output_base / "basins_passed_custom.csv"
#             gage_list_path = sandbox_dir / "basin_IDs" / "basin_IDs.csv"

#             if passed_basins_csv.exists() and gage_list_path.exists():
#                 try:
#                     # Read gage IDs from provided list
#                     gage_ids_df = pd.read_csv(gage_list_path, dtype=str)
#                     gage_ids = set(gage_ids_df['gage_id'].astype(str).str.strip())

#                     # Read basins_passed.csv
#                     basins_df = pd.read_csv(passed_basins_csv, dtype=str)

#                     # Determine matching column name in basins_passed.csv
#                     possible_id_cols = ["STAID", "gage_id", "gageID", "id"]
#                     matching_col = next((col for col in possible_id_cols if col in basins_df.columns), None)

#                     if matching_col is None:
#                         print(f"Warning: No matching gage ID column found in {passed_basins_csv}")
#                     else:
#                         filtered_df = basins_df[basins_df[matching_col].astype(str).isin(gage_ids)]
#                         filtered_df.to_csv(custom_csv, index=False)
#                         print(f"Filtered basins_passed_custom.csv created at {custom_csv} with {len(filtered_df)} basins.")
#                 except Exception as e:
#                     print(f"Warning: Failed to create filtered basins_passed_custom.csv - {e}")
#             else:
#                 if not passed_basins_csv.exists():
#                     print(f"Warning: basins_passed.csv not found at {passed_basins_csv}")
#                 if not gage_list_path.exists():
#                     print(f"Warning: basin_IDs.csv not found at {gage_list_path}")

#     if (args.run):
#         print ("Calling Runner...")

#         # Pass particle selection through environment so Runner can opt-in without API breakage
#         if getattr(args, "concurrent_particles", False):
#             os.environ["NGEN_CONCURRENT_PARTICLES"] = "1"
#         if getattr(args, "particle_id", None) is not None:
#             os.environ["NGEN_PARTICLE_ID"] = str(args.particle_id)

#         _runner = runner.Runner(sandbox_config, calib_config, gage_id=args.gage_id)
#         status  = _runner.run()

#         if (status):
#             sys.exit("Failed during ngen-cal execution...")
#         else:
#             print ("DONE \u2713")

#     print ("**********************************")


# if __name__ == "__main__":

#     try:
#         parser = argparse.ArgumentParser()
#         parser.add_argument("-subset", action='store_true',    help="Subset basin (generate .gpkg files)")
#         parser.add_argument("-forc",   action='store_true',    help="Download forcing data")
#         parser.add_argument("-conf",   action='store_true',    help="Generate config files")
#         parser.add_argument("-run",    action='store_true',    help="Run NextGen simulations")
#         parser.add_argument("-i",      dest="sandbox_infile", type=str, required=False,  help="sandbox config file")
#         parser.add_argument("-j",      dest="calib_infile",    type=str, required=False,  help="caliberation config file")
#         parser.add_argument("--gage_id", type=str, required=False, help="Run model only for this gage ID")
#         # New concurrent-eval flags
#         parser.add_argument("--concurrent-particles", action="store_true",
#                             help="Enable per-particle workspaces for concurrent PSO evaluation")
#         parser.add_argument("--num-particles", type=int, default=1,
#                             help="Number of particle workspaces to scaffold at -conf")
#         parser.add_argument("--particle-id", type=int, default=None,
#                             help="Target particle workspace when running (-run)")
#         args = parser.parse_args()
#     except SystemExit:
#         print("Formulations supported:\n" + "\n".join(formulations_supported))
#         sys.exit(0)

#     if (args.sandbox_infile):
#         if (os.path.exists(args.sandbox_infile)):
#             sandbox_config = Path(args.sandbox_infile).resolve()
#         else:
#             print ("sandbox config file DOES NOT EXIST, provided: ", args.sandbox_infile)
#             sys.exit(0)
#     else:
#         sandbox_config = f"{sandbox_dir}/configs/sandbox_config.yaml"

#     if (args.calib_infile):
#         if (os.path.exists(args.calib_infile)):
#             calib_config = Path(args.calib_infile).resolve()
#         else:
#             print ("caliberation config file DOES NOT EXIST, provided: ", args.calib_infile)
#             sys.exit(0)
#     else:
#         calib_config = f"{sandbox_dir}/configs/calib_config.yaml"

#     if (len(sys.argv) < 2):
#         print ("No arguments are provide")
#         sys.exit(0)

#     # check if expected Python virtual env exists and activated
#     CheckSandbox_VENV()

#     Sandbox(sandbox_config, calib_config)

































# ############################################################################################
# # Author  : Ahmad Jan Khattak
# # Contact : ahmad.jan.khattak@noaa.gov
# # Date    : July 16, 2024
# ############################################################################################
# ###############################################################
# # edits from: Peter La Follette [plafollette@lynker.com | May 2025], to allow for tiled formulations

# import os, sys
# import subprocess
# import yaml
# import argparse
# from pathlib import Path
# import pandas as pd

# path = Path(sys.argv[0]).resolve()
# sandbox_dir = path.parent

# from src.python import forcing, driver, runner


# def CheckSandbox_VENV():
#     VENV_SANDBOX = Path.home() / ".venv_sandbox_py3.11"

#     # Check if the virtual environment exists
#     if not VENV_SANDBOX.exists():
#         print(f"Error: NextGen virtual environment {VENV_SANDBOX} not found under home directory...")
#         sys.exit(1)


#     # Check if the script is running inside that required environment
#     VENV_ACTIVE = Path(sys.prefix)
#     if VENV_ACTIVE.resolve() != VENV_SANDBOX.resolve():
#         print(f"Warning: sandbox.py is not running in the expected Python virtual environment.")
#         print(f"Expected: {VENV_SANDBOX}")
#         print(f"Active:   {VENV_ACTIVE}")

#         sys.exit(1)


# formulations_supported = [
#     "NOM,CFE",
#     "PET,CFE",
#     "NOM,LASAM",
#     "PET,LASAM",
#     "NOM,CFE,PET",
#     "NOM,CFE,SMP,SFT",
#     "NOM,LASAM,SMP,SFT",
#     "NOM,TOPMODEL",
#     "BASELINE,CFE",
#     "BASELINE,LAS"
# ]

# def Sandbox(sandbox_config, calib_config):
    
#     if (args.subset):
#         print ("Generating geopackages...")
#         subset_basin = f"Rscript {sandbox_dir}/src/R/main.R {sandbox_config}"
#         status = subprocess.call(subset_basin,shell=True)

#         if (status):
#             sys.exit("Failed during generating geopackge(s) step...")
#         else:
#             print ("DONE \u2713")

#     if (args.forc):
#         print ("Generating forcing data...")
#         process_forcing = forcing.ForcingProcessor(sandbox_config)
#         status          = process_forcing.download_forcing()

#         if (status):
#             sys.exit("Failed during generating geopackge(s) step...")
#         else:
#             print ("DONE \u2713")

#     if (args.conf):
#         print ("Generating config files...")
#         _driver = driver.Driver(sandbox_config, formulations_supported)
#         status  = _driver.run()

#         if (status):
#             sys.exit("Failed during generating config files step...")
#         else:
#             print ("DONE \u2713")

#             # Disable Spotlight indexing in outputs/div and troute if they exist or might be used
#             import platform
#             from pathlib import Path
#             import glob
#             import yaml

#             def disable_spotlight_indexing(path_str):
#                 path = Path(path_str)
#                 if platform.system() == "Darwin":
#                     path.mkdir(parents=True, exist_ok=True)
#                     (path / ".metadata_never_index").touch()

#             # Parse output_dir from the sandbox config file
#             with open(sandbox_config, "r") as f:
#                 sandbox_dict = yaml.safe_load(f)
#             output_base = Path(sandbox_dict["output_dir"])

#             # Create postproc directory at same level as output_dir
#             postproc_dir = output_base.parent / "postproc"
#             postproc_dir.mkdir(parents=True, exist_ok=True)
#             disable_spotlight_indexing(postproc_dir)

#             # Apply to all gage output dirs created
#             gage_dirs = glob.glob(str(output_base / "*" / "outputs" / "div"))
#             for div_path in gage_dirs:
#                 disable_spotlight_indexing(div_path)

#             for gage_path in output_base.glob("*"):
#                 if not gage_path.is_dir():
#                     continue  # skip files like .DS_Store
#                 div_path = gage_path / "outputs" / "div"
#                 troute_path = gage_path / "troute"

#                 disable_spotlight_indexing(div_path)

#                 if not troute_path.exists():
#                     try:
#                         troute_path.mkdir(parents=True, exist_ok=True)
#                     except Exception as e:
#                         print(f"Warning: Could not create troute dir: {troute_path} - {e}")
#                 disable_spotlight_indexing(troute_path)


#             # Create a filtered basins_passed_custom.csv file
#             import pandas as pd

#             passed_basins_csv = output_base / "basins_passed.csv"
#             custom_csv = output_base / "basins_passed_custom.csv"
#             gage_list_path = sandbox_dir / "basin_IDs" / "basin_IDs.csv"

#             if passed_basins_csv.exists() and gage_list_path.exists():
#                 try:
#                     # Read gage IDs from provided list
#                     gage_ids_df = pd.read_csv(gage_list_path, dtype=str)
#                     gage_ids = set(gage_ids_df['gage_id'].astype(str).str.strip())

#                     # Read basins_passed.csv
#                     basins_df = pd.read_csv(passed_basins_csv, dtype=str)

#                     # Determine matching column name in basins_passed.csv
#                     possible_id_cols = ["STAID", "gage_id", "gageID", "id"]
#                     matching_col = next((col for col in possible_id_cols if col in basins_df.columns), None)

#                     if matching_col is None:
#                         print(f"Warning: No matching gage ID column found in {passed_basins_csv}")
#                     else:
#                         filtered_df = basins_df[basins_df[matching_col].astype(str).isin(gage_ids)]
#                         filtered_df.to_csv(custom_csv, index=False)
#                         print(f"Filtered basins_passed_custom.csv created at {custom_csv} with {len(filtered_df)} basins.")
#                 except Exception as e:
#                     print(f"Warning: Failed to create filtered basins_passed_custom.csv - {e}")
#             else:
#                 if not passed_basins_csv.exists():
#                     print(f"Warning: basins_passed.csv not found at {passed_basins_csv}")
#                 if not gage_list_path.exists():
#                     print(f"Warning: basin_IDs.csv not found at {gage_list_path}")


        
#     if (args.run):
#         print ("Calling Runner...")

#         _runner = runner.Runner(sandbox_config, calib_config, gage_id=args.gage_id)
#         status  = _runner.run()

#         if (status):
#             sys.exit("Failed during ngen-cal execution...")
#         else:
#             print ("DONE \u2713")
    
#     print ("**********************************")
    
    

# if __name__ == "__main__":
    
#     try:
#         parser = argparse.ArgumentParser()
#         parser.add_argument("-subset", action='store_true',    help="Subset basin (generate .gpkg files)")
#         parser.add_argument("-forc",   action='store_true',    help="Download forcing data")
#         parser.add_argument("-conf",   action='store_true',    help="Generate config files")
#         parser.add_argument("-run",    action='store_true',    help="Run NextGen simulations")
#         parser.add_argument("-i",      dest="sandbox_infile", type=str, required=False,  help="sandbox config file")
#         parser.add_argument("-j",      dest="calib_infile",    type=str, required=False,  help="caliberation config file")
#         parser.add_argument("--gage_id", type=str, required=False, help="Run model only for this gage ID")
#         args = parser.parse_args()
#     except SystemExit:
#         print("Formulations supported:\n" + "\n".join(formulations_supported))
#         sys.exit(0)

#     if (args.sandbox_infile):
#         if (os.path.exists(args.sandbox_infile)):
#             sandbox_config = Path(args.sandbox_infile).resolve()
#         else:
#             print ("sandbox config file DOES NOT EXIST, provided: ", args.sandbox_infile)
#             sys.exit(0)
#     else:
#         sandbox_config = f"{sandbox_dir}/configs/sandbox_config.yaml"

#     if (args.calib_infile):
#         if (os.path.exists(args.calib_infile)):
#             calib_config = Path(args.calib_infile).resolve()
#         else:
#             print ("caliberation config file DOES NOT EXIST, provided: ", args.calib_infile)
#             sys.exit(0)
#     else:
#         calib_config = f"{sandbox_dir}/configs/calib_config.yaml"
    
#     if (len(sys.argv) < 2):
#         print ("No arguments are provide")
#         sys.exit(0)

#     # check if expected Python virtual env exists and activated
#     CheckSandbox_VENV()

#     Sandbox(sandbox_config, calib_config)
