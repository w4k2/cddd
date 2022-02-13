import strlearn as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from methods import CLFD
from methods import CentroidDistanceDriftDetector
from methods import CentroidDistanceDriftDetectorV2
from methods import D3

from skmultiflow import drift_detection

from sklearn.neural_network import MLPClassifier
from skmultiflow.bayes import NaiveBayes
from sklearn.base import clone

from joblib import Parallel, delayed
from time import time

import logging
import traceback
import warnings
import os

warnings.simplefilter("ignore")


logging.basicConfig(filename='experiment_main.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("")
logging.info("----------------------")
logging.info("NEW EXPERIMENT STARTED")
logging.info(os.path.basename(__file__))
logging.info("----------------------")
logging.info("")


def compute(clf_name, clf, chunk_size, n_chunks, experiment_name, stream_name):

    # print(clf_name, clf, chunk_size, n_chunks, experiment_name, stream_name)

    logging.basicConfig(filename='experiment_main.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        warnings.filterwarnings("ignore")

        drift = stream_name.split("/")[-2]

        if os.path.exists("results/raw_conf/%s/%s/%s/%s.csv" % (experiment_name, drift, stream_name.split("/")[-1][0:-5], clf_name)):
            return

        print("START: %s, %s, %s" % (drift, stream_name, clf_name))
        logging.info("START - %s, %s, %s" % (drift, stream_name,  clf_name))
        start = time()

        stream = sl.streams.ARFFParser(stream_name, chunk_size, n_chunks)

        evaluator = sl.evaluators.TestThenTrain(verbose=False)
        evaluator.process(stream, clone(clf))

        stream_name = stream_name.split("/")[-1][0:-5]

        filename = "results/raw_conf/%s/%s/%s/%s.csv" % (experiment_name, drift, stream_name, clf_name)
        if not os.path.exists("results/raw_conf/%s/%s/%s/" % (experiment_name, drift, stream_name)):
            os.makedirs("results/raw_conf/%s/%s/%s/" % (experiment_name, drift, stream_name))
        np.savetxt(fname=filename, fmt="%d, %d, %d, %d", X=evaluator.confusion_matrix[0])

        drifts = evaluator.clfs_[0].drifts

        filename = "results/raw_conf/%s/%s/%s/%s_drifts.csv" % (experiment_name, drift, stream_name, clf_name)
        if not os.path.exists("results/raw_conf/%s/%s/%s/" % (experiment_name, drift, stream_name)):
            os.makedirs("results/raw_conf/%s/%s/%s/" % (experiment_name, drift, stream_name))
        np.savetxt(fname=filename, fmt="%d", X=drifts)

        if clf_name == "CDDD":
            dd = evaluator.clfs_[0].drift_detector
            distances = cdist(dd.centroids[0], dd.centroids[0], "cityblock")

            fig, ax1 = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

            ax1.imshow(distances, cmap="gist_gray")
            ax1.set_title("Distance matrix")
            ax1.set_ylabel("Data chunk")
            ax1.set_xlabel("Data chunk")
            ax1.set_xlim(0, stream.n_chunks)
            ax1.set_ylim(stream.n_chunks, 0)

            for idx in dd.concepts[0]:
                ax1.axvline(idx, 0, stream.n_chunks, linestyle="-.", linewidth=1.5, color="tab:blue")

            filename = "results/plots/%s/matrix/%s_monly" % (experiment_name, stream_name)
            if not os.path.exists("results/plots/%s/matrix/" % (experiment_name)):
                os.makedirs("results/plots/%s/matrix/" % (experiment_name))
            fig.savefig(filename+".png", bbox_inches='tight')
            plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
            plt.close()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7.5), gridspec_kw={
                'height_ratios': [3, 1]}, constrained_layout=True)

            ax1.imshow(distances, cmap="gist_gray")
            ax1.set_title("Distance matrix")
            ax1.set_ylabel("Data chunk")
            ax1.set_xlabel("Data chunk")
            ax1.set_xlim(0, stream.n_chunks)
            ax1.set_ylim(stream.n_chunks, 0)

            for idx in dd.concepts[0]:
                ax1.axvline(idx, 0, stream.n_chunks, linestyle="-.", linewidth=1.5, color="tab:blue")

            ax2.plot(range(stream.n_chunks), dd.mean_distances[0], linewidth=1.5, color="black")
            ax2.plot(range(1, stream.n_chunks), dd.con_array[0], linestyle="--", linewidth=1, color="tab:orange")
            ax2.set_xlim(0, stream.n_chunks)
            ax2.set_xlabel("Data chunk")
            ax2.set_ylabel("Metric")

            for idx in dd.concepts[0]:
                ax2.axvline(idx, 0, 1, linestyle="-.", linewidth=1, color="tab:blue")

            # fig.suptitle("Centroid distance drift detection", fontsize=16)
            filename = "results/plots/%s/matrix/%s_m" % (experiment_name, stream_name)
            if not os.path.exists("results/plots/%s/matrix/" % (experiment_name)):
                os.makedirs("results/plots/%s/matrix/" % (experiment_name))
            fig.savefig(filename+".png", bbox_inches='tight')
            plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
            plt.close()

        end = time()-start

        print("DONE: %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, end))
        logging.info("DONE - %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, end))

    except Exception as ex:
        logging.exception("Exception in %s, %s, %s" % (drift, stream_name, clf_name))
        print("ERROR: %s, %s, %s" % (drift, stream_name, clf_name))
        traceback.print_exc()
        print(str(ex))


streams = []  # name                                                             size    n_chunks
streams.append(("streams/real/1-2-3-4-5vsA_penbased.arff", 500, int(10992/500)))
streams.append(("streams/real/1-5vsA_INSECTS-abrupt_imbalanced_norm.arff", 500, int(355275/500)))
streams.append(("streams/real/1-5vsA_INSECTS-incremental-abrupt_imbalanced_norm.arff", 500, int(452044/500)))
streams.append(("streams/real/1-5vsA_INSECTS-incremental_imbalanced_norm.arff", 500, int(452044/500)))
streams.append(("streams/real/1vsA_shuttle.arff", 500, int(57999/500)))
streams.append(("streams/real/2-5vsA_INSECTS-gradual_imbalanced_norm.arff", 500, int(143323/500)))
streams.append(("streams/real/2vsA_covtypeNorm.arff", 500, int(581012/500)))
streams.append(("streams/real/elecNormNew.arff", 500, int(45312/500)))
streams.append(("streams/real/magic_shuffled.arff", 500, int(19020/500)))
streams.append(("streams/real/NOAA.arff", 500, int(18159/500)))


experiments = {
                "mlp": MLPClassifier((10, 10), random_state=123),
                "nby": NaiveBayes(),
              }

experiment_names = list(experiments.keys())
base_estimators = list(experiments.values())


for estimator, experiment_name in zip(base_estimators, experiment_names):

    methods = {
                "CDDD": CLFD(estimator=estimator, drift_detector=CentroidDistanceDriftDetector()),
                "CDDD-V2": CLFD(estimator=estimator, drift_detector=CentroidDistanceDriftDetectorV2()),
                "ADWIN": CLFD(estimator=estimator, drift_detector=drift_detection.ADWIN()),
                "EDDM": CLFD(estimator=estimator, drift_detector=drift_detection.EDDM()),
                "D3": CLFD(estimator=estimator, drift_detector=D3()),
                "BASE": CLFD(estimator=estimator, drift_detector=None),
                "DDM": CLFD(estimator=estimator, drift_detector=drift_detection.DDM(out_control_level=1.5, min_num_instances=30)),
              }

    clfs = list(methods.values())
    clf_names = list(methods.keys())

    Parallel(n_jobs=-1)(delayed(compute)
                        (clf_name, clf, chunk_size, n_chunks, experiment_name, stream_name)
                        for (clf_name, clf) in zip(clf_names, clfs)
                        for (stream_name, chunk_size, n_chunks) in streams
                        )

logging.info("-------------------")
logging.info("EXPERIMENT FINISHED")
logging.info("-------------------")
