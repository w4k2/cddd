import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcdefaults
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


def plot_streams_matplotlib(methods, streams, metrics, experiment_name, gauss=0, methods_alias=None, metrics_alias=None):
    rcdefaults()

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = {}

    for stream_name in streams:
        for clf_name in methods:
            for metric in metrics:
                try:
                    filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                    data[stream_name, clf_name, metric] = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                except Exception:
                    data[stream_name, clf_name, metric] = None
                    print("Error in loading data", stream_name, clf_name, metric)

    # styles = ["--", "--", "--", "--", "--", "-"]
    # colors = ["black", "tab:red", "tab:orange", "tab:cyan", "tab:blue", "tab:green"]

    styles = ['--', '--', '--', '--', '--', '--', '--', '--']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray']
    widths = [1.5, 1, 1, 1, 1, 1, 1, 1, 1]

    for stream_name in tqdm(streams, "Plotting %s" % experiment_name):
        for metric, metric_a in zip(metrics, metrics_alias):

            for idx, (clf_name, method_a) in reversed(list(enumerate(zip(methods, methods_alias)))):
                if data[stream_name, clf_name, metric] is None:
                    continue

                plot_data = data[stream_name, clf_name, metric]

                if gauss > 0:
                    plot_data = gaussian_filter1d(plot_data, gauss)

                if clf_name == "BASE":
                    plt.plot(range(len(plot_data)), plot_data, label=method_a, linestyle="-", color="black", linewidth=0.75)
                else:
                    if colors is None:
                        plt.plot(range(len(plot_data)), plot_data, label=method_a)
                    else:
                        plt.plot(range(len(plot_data)), plot_data, label=method_a, linestyle=styles[idx], color=colors[idx], linewidth=widths[idx])

            filename = "results/plots/%s/%s/%s" % (experiment_name, metric, stream_name)
            stream_name_ = "/".join(stream_name.split("/")[0:-1])
            if not os.path.exists("results/plots/%s/%s/%s/" % (experiment_name, metric, stream_name_)):
                os.makedirs("results/plots/%s/%s/%s/" % (experiment_name, metric, stream_name_))

            plt.legend()
            plt.legend(reversed(plt.legend().legendHandles), methods_alias, loc="lower center", ncol=len(methods_alias))
            # plt.title(metric_a+"     "+experiment_name+"     "+stream_name_2)
            plt.ylabel(metric_a)
            plt.ylim(0, 1)
            plt.xlim(0, len(plot_data)-1)
            plt.xlabel("Data chunk")
            plt.gcf().set_size_inches(5, 5)
            plt.grid(True, color="silver", linestyle=":")
            plt.savefig(filename+".png", bbox_inches='tight')
            plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
            plt.clf()
            plt.close()
