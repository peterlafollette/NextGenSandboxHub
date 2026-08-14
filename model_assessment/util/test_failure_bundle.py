import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_assessment.util.failure_bundle import capture_failure_bundle, env_flag


class FailureBundleTest(unittest.TestCase):
    def test_capture_is_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(env_flag("CAPTURE_FAILURE_BUNDLES"))
        with patch.dict(os.environ, {"CAPTURE_FAILURE_BUNDLES": "true"}, clear=True):
            self.assertTrue(env_flag("CAPTURE_FAILURE_BUNDLES"))

    def test_capture_snapshots_exact_particle_inputs_without_copying_forcing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "in"
            gage_input = input_root / "01196500" / "data"
            gage_input.mkdir(parents=True)
            gpkg = gage_input / "catchment_data.gpkg"
            forcing = gage_input / "forcing.nc"
            gpkg.write_bytes(b"gpkg")
            forcing.write_bytes(b"forcing")

            work = root / "work"
            (work / "configs" / "casam").mkdir(parents=True)
            (work / "json").mkdir()
            (work / "outputs" / "div").mkdir(parents=True)
            soil = work / "configs" / "casam" / "soil_params.txt"
            soil.write_text("soil parameters\n")
            casam_config = work / "configs" / "casam" / "casam_cfg_cat-1.txt"
            casam_config.write_text(f"a=0.1\nsoil_params_file={soil}\n")
            (work / "outputs" / "div" / "cat-1.csv").write_text("time,value\n1,2\n")
            realization = work / "json" / "realization.json"
            realization.write_text(
                json.dumps(
                    {
                        "time": {"start_time": "2010-01-01 00:00:00", "end_time": "2011-01-01 00:00:00"},
                        "global": {
                            "forcing": {"path": str(forcing)},
                            "formulations": [
                                {
                                    "name": "bmi_multi",
                                    "params": {
                                        "model_type_name": "PET_CASAM",
                                        "modules": [
                                            {
                                                "name": "bmi_c",
                                                "params": {
                                                    "model_type_name": "CASAM",
                                                    "library_file": "/hpc/LGAR-C/build/liblasambmi.so",
                                                    "init_config": str(casam_config),
                                                },
                                            }
                                        ],
                                    },
                                }
                            ],
                        },
                    }
                )
            )

            run_log = root / "ngen.log"
            run_log.write_text("assertion failed\n")
            metadata = root / "ngen_run_metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "returncode": 134,
                        "signal": 6,
                        "gpkg_file": str(gpkg),
                        "realization_path": str(realization),
                        "ngen_executable": "/missing/ngen",
                    }
                )
            )
            sandbox_config = root / "sandbox.yaml"
            sandbox_config.write_text("simulation: {}\n")

            project = Path(__file__).resolve().parents[2]
            with patch.dict(os.environ, {"CIROH_INPUT_DIR": str(input_root)}, clear=True):
                bundle = capture_failure_bundle(
                    destination_root=root / "bundles",
                    gage_id="01196500",
                    iteration=3,
                    particle_idx=4,
                    tile_idx=0,
                    stage="objective",
                    work_root=work,
                    realization_path=realization,
                    params=[1.25, 2.5],
                    param_names=["alpha", "n"],
                    sandbox_config=sandbox_config,
                    sandbox_returncode=1,
                    ngen_log_path=run_log,
                    ngen_metadata_path=metadata,
                    project_root=project,
                    include_forcing=False,
                )

            self.assertTrue((bundle / "particle_workspace" / "json" / "realization.json").is_file())
            self.assertTrue((bundle / "input" / "geopackage" / gpkg.name).is_file())
            self.assertFalse((bundle / "input" / "forcing" / forcing.name).exists())
            self.assertTrue((bundle / "replay_ngen_failure.py").is_file())
            self.assertEqual(json.loads((bundle / "parameters.json").read_text())["by_name"]["n"], 2.5)
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(manifest["ngen_returncode"], 134)
            self.assertEqual(manifest["ngen_signal"], 6)
            references = json.loads((bundle / "input_references.json").read_text())
            forcing_reference = next(item for item in references if item["original_path"] == str(forcing.resolve()))
            self.assertEqual(forcing_reference["relative_to_input_root"], "01196500/data/forcing.nc")
            self.assertEqual(forcing_reference["sha256"], "feb25aff76d880178f2b3d1521f2e56bdd8b0ed403375248d66f0c6cf1d904a0")
            self.assertFalse(any(path.name.endswith(".tmp") for path in bundle.parent.iterdir()))

            lgar_repo = root / "LGAR-C"
            (lgar_repo / "build").mkdir(parents=True)
            (lgar_repo / "build" / "liblasambmi.so").write_bytes(b"library")
            fake_ngen = root / "ngen"
            fake_ngen.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "realization = json.loads(pathlib.Path(sys.argv[5]).read_text())\n"
                "assert 'output_root' in realization and 'output_root' not in realization['global']\n"
                "module = realization['global']['formulations'][0]['params']['modules'][0]['params']\n"
                "assert pathlib.Path(module['library_file']).is_file()\n"
                "config = pathlib.Path(module['init_config'])\n"
                "assert config.is_file()\n"
                "soil_path = config.read_text().split('soil_params_file=', 1)[1].splitlines()[0]\n"
                "assert pathlib.Path(soil_path).is_file()\n"
                "assert pathlib.Path(realization['global']['forcing']['path']).is_file()\n"
            )
            fake_ngen.chmod(0o755)
            shutil.rmtree(work)
            replay = subprocess.run(
                [
                    sys.executable,
                    str(bundle / "replay_ngen_failure.py"),
                    "--ngen",
                    str(fake_ngen),
                    "--lgarto-repo",
                    str(lgar_repo),
                    "--input-root",
                    str(input_root),
                ],
                env={**os.environ, "TMPDIR": str(root)},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(replay.returncode, 0, replay.stdout)


if __name__ == "__main__":
    unittest.main()
