import os
import numpy as np
from collections import defaultdict

"""This script was AI generated to parse log files and extract performance metrics"""

BASE_DIR = os.path.expanduser("~/Buffalo/Online/3")
# SUBFOLDERS = ["0.2", "0.4", "0.6", "0.8", "None"]
# SUBFOLDERS = ["0.2", "0.4", "0.6", "0.7", "None"]
SUBFOLDERS = ["-0.3", "-0.1", "0.1", "0.3", "None"]
# SUBFOLDERS = ["0.5", "0.75", "1.0", "1.25", "None"]
# BASE_DIR = os.path.expanduser("~/Buffalo/Offline/β Test")
# SUBFOLDERS = ["1/5", "2/5", "3/5"]

# data[method][metric] -> list of values
data = defaultdict(lambda: defaultdict(list))


def parse_log(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    current_method = None

    for line in lines:
        if line == "Experiments":
            continue

        if (
            line.startswith("PAR")
            or line == "A2P"
            or line == "NRMDP"
        ):
            current_method = line
            continue

        if ":" in line and current_method is not None:
            metric, value = line.split(":")
            metric = metric.strip()
            value = float(value.strip())
            data[current_method][metric].append(value)


# ---- parse all files ----
for sub in SUBFOLDERS:
    path = os.path.join(BASE_DIR, sub, "log.txt")
    parse_log(path)


# ---- write output ----
out_path = os.path.join(BASE_DIR, "log.txt")

with open(out_path, "w") as f:
    f.write("Experiments\n")

    for method in data:
        f.write(f"{method}\n")

        for metric in data[method]:
            values = np.array(data[method][metric])

            mean = np.mean(values)
            std = np.std(values)

            f.write(f"{metric}: {mean:.6f} ± {std:.6f}\n")