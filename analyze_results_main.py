from core import calculate_metrics
from core import plot_streams_matplotlib
from core import pairs_metrics_multi

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

stream_sets = []
streams_aliases = []

# -------------------------------------------------------------------

streams = []
for i in range(1, 11):
    directory = f"sl_{i}d/incremental/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

for i in range(1, 11):
    directory = f"mf_{i}d/incremental/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

for i in range(1, 11):
    directory = f"rv_{i}d/incremental/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["incremental"]

# -------------------------------------------------------------------

streams = []
for i in range(1, 11):
    directory = f"sl_{i}d/sudden/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

for i in range(1, 11):
    directory = f"mf_{i}d/sudden/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

for i in range(1, 11):
    directory = f"rv_{i}d/sudden/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["sudden"]

# -------------------------------------------------------------------

streams = []
for i in range(1, 11):
    directory = f"sl_{i}d/gradual/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

for i in range(1, 11):
    directory = f"mf_{i}d/gradual/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

for i in range(1, 11):
    directory = f"rv_{i}d/gradual/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["gradual"]

# -------------------------------------------------------------------

streams = []
for i in range(1, 11):
    directory = f"sl_{i}d/recurring/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

for i in range(1, 11):
    directory = f"mf_{i}d/recurring/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

for i in range(1, 11):
    directory = f"rv_{i}d/recurring/"
    mypath = "results/raw_conf/mlp/%s" % directory
    streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["recurring"]

# -------------------------------------------------------------------

streams = []
directory = "real/"
mypath = "results/raw_conf/mlp/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["real"]

# -------------------------------------------------------------------

method_names = [
                "CDDD",
                "CDDD-V2",
                "ADWIN",
                "DDM",
                "EDDM",
                "D3",
                "BASE",
              ]

methods_alias = [
                "CDDD",
                "CDDD-V2",
                "ADWIN",
                "DDM",
                "EDDM",
                "D3",
                "BASE",
                ]

metrics_alias = [
           "Accuracy",
           "Gmean",
           "F-score",
           "recall",
           "specificity",

          ]

metrics = [
           "accuracy",
           "g_mean",
           "f1_score",
           "recall",
           "specificity",

          ]


experiment_names = [
                    "mlp",
                    "nby",
                    ]

for streams, streams_alias in zip(stream_sets, streams_aliases):
    for experiment_name in experiment_names:
        calculate_metrics(method_names, streams, metrics, experiment_name, recount=True)
        plot_streams_matplotlib(method_names, streams, metrics, experiment_name, gauss=5, methods_alias=methods_alias, metrics_alias=metrics_alias)

    pairs_metrics_multi(method_names, streams, metrics, experiment_names, methods_alias=methods_alias, metrics_alias=metrics_alias, streams_alias=streams_alias, title=False)
