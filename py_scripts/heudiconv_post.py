import json
from thalpy import base
from thalpy.constants import paths
import glob as glob
import sys
import os
import stat

sub_args = sys.argv[1:]
sub = sub_args[0]
dataset_dir = sub_args[1]

fieldmap_jsons = glob.glob(dataset_dir + "/BIDS/sub-" + sub + "/fmap/*.json")
runs = sorted(glob.glob(dataset_dir + "/BIDS/sub-" + sub + "/func/*.nii.gz"))
runs = [run.split(f"sub-{sub}/")[1] for run in runs]
print(runs)
for fmap in fieldmap_jsons:
    os.chmod(fmap, 0o755)

    with open(fmap, "r") as jsonfile:
        json_content = json.load(jsonfile)

    if base.parse_run_from_file(fmap) == "001":
        intended_runs = runs[:3] # first 3 runs get assigned to fieldmap 1
    elif base.parse_run_from_file(fmap) == "002":
        intended_runs = runs[3:] # remaining runs get assigned to fieldmap 2
    else:
        print("Run number must be 001 or 002")
        continue

    json_content["IntendedFor"] = intended_runs
    # correct json file intended_runs

    with open(fmap, "w") as jsonfile:
        # you decide the indentation level
        json.dump(json_content, jsonfile, indent=4)
