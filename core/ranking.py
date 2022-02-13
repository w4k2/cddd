from scipy import stats
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import sqrt, ceil


def pairs_metrics_multi(method_names, stream_names, metrics, experiment_names, methods_alias=None, streams_alias=None, metrics_alias=None, treshold=0.05, title=False):
    if metrics_alias is None:
        metrics_alias = metrics
    if methods_alias is None:
        methods_alias = method_names
    if streams_alias is None:
        streams_alias = stream_names[0].split("/")[0]

    # --------------------------------------
    # Load data
    # --------------------------------------
    data = {}
    for method_name in method_names:
        for stream_name in stream_names:
            for metric in metrics:
                for experiment_name in experiment_names:
                    try:
                        data[(method_name, stream_name, metric, experiment_name)] = np.genfromtxt("results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, method_name))
                    except:
                        print("None is ", method_name, stream_name, metric, experiment_name)
                        data[(method_name, stream_name, metric, experiment_name)] = None
                        print(data[(method_name, stream_name, metric, experiment_name)])

    plt.rc('ytick', labelsize=12)
    fig, axes = plt.subplots(len(experiment_names), len(metrics))
    fig.subplots_adjust(wspace=0.6, hspace=0.2)
    if title:
        fig.suptitle(methods_alias[0]+" "+streams_alias, fontsize=32, x=0.5, y=1.05)

    # --------------------------------------
    # Init/clear ranks
    # --------------------------------------
    for index_i, experiment_name in enumerate(experiment_names):
        for index_j, (metric, metric_a) in enumerate(zip(metrics, metrics_alias)):
            ranking = {}
            for method_name in method_names:
                ranking[method_name] = {"win": 0, "lose": 0, "tie": 0, "error": 0}

            # --------------------------------------
            # Pair tests
            # --------------------------------------
            for stream in tqdm(stream_names, "Rank %s %s" % (experiment_name, metric)):
                method_1 = method_names[0]
                for j, method_2 in enumerate(method_names):
                    if method_1 == method_2:
                        continue
                    # if data[(method_2, stream, metric, experiment_name)] is None:
                    #     print("None data", method_2, stream, experiment_name)
                    #     continue
                    # if data[(method_1, stream, metric, experiment_name)] is None:
                    #     print("None data", method_1, stream, experiment_name)
                    #     continue

                    try:
                        statistic, p_value = stats.ranksums(data[(method_1, stream, metric, experiment_name)], data[(method_2, stream, metric, experiment_name)])
                        if p_value < treshold:
                            if statistic > 0:
                                ranking[method_2]["win"] += 1
                                # print("WIN", method_2, stream)
                            else:
                                ranking[method_2]["lose"] += 1
                                print("LOSE", method_2, stream)
                        else:
                            ranking[method_2]["tie"] += 1
                            # print("TIE", method_2, stream)
                    except:
                        ranking[method_2]["error"] += 1
                        print("Exception", method_1, method_2, stream, metric, experiment_name)

            # --------------------------------------
            # Count ranks
            # --------------------------------------
            rank_win = []
            rank_tie = []
            rank_lose = []
            rank_error = []

            for method_name in method_names[1:]:
                rank_win.append(ranking[method_name]['win'])
                rank_tie.append(ranking[method_name]['tie'])
                rank_lose.append(ranking[method_name]['lose'])
                try:
                    rank_error.append(ranking[method_name]['error'])
                except Exception:
                    pass

            rank_win.reverse()
            rank_tie.reverse()
            rank_lose.reverse()
            rank_error.reverse()

            rank_win = np.array(rank_win)
            rank_tie = np.array(rank_tie)
            rank_lose = np.array(rank_lose)
            rank_error = np.array(rank_error)
            ma = methods_alias[1:].copy()
            ma.reverse()

            # --------------------------------------
            # Plotting
            # --------------------------------------

            axes[index_i, index_j].barh(ma, rank_win, color="green", height=0.9)
            axes[index_i, index_j].barh(ma, rank_tie, left=rank_win, color="gold", height=0.9)
            axes[index_i, index_j].barh(ma, rank_lose, left=rank_win+rank_tie, color="crimson", height=0.9)
            try:
                axes[index_i, index_j].barh(ma, rank_error, left=rank_win+rank_tie+rank_lose, color="crimson", height=0.9)
            except Exception:
                pass
            axes[index_i, index_j].set_xlim([0, len(stream_names)])
            N_of_streams = len(stream_names)
            critical_difference = ceil(N_of_streams/2 + 1.96*sqrt(N_of_streams)/2)
            if len(stream_names) < 25:
                axes[index_i, index_j].axvline(critical_difference, 0, 1, linestyle="--", linewidth=3, color="red")
            else:
                axes[index_i, index_j].axvline(critical_difference, 0, 1, linestyle="--", linewidth=3, color="black")

    for i, experiment_name in enumerate(experiment_names):
        axes[i, 0].set_ylabel(experiment_name.upper(), rotation=0, fontsize=22)
        axes[i, 0].get_yaxis().set_label_coords(-0.7, 0.4)

    for j, metric_a in enumerate(metrics_alias):
        axes[0, j].set_title(metric_a.upper(), fontsize=22)

    if not os.path.exists("results/ranking_plots/%s/" % (experiment_name)):
        os.makedirs("results/ranking_plots/%s/" % (experiment_name))
    plt.gcf().set_size_inches(15, 5)
    filename = "results/ranking_plots/ranks_%s_%s" % (streams_alias, methods_alias[0])
    # if title:
    #     filename = "results/ranking_plots/ranks_%s_%s" % (streams_alias, methods_alias[0])
    # else:
    #     filename = "results/ranking_plots/ranks_%s" % (streams_alias)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
    plt.clf()
