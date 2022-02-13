import strlearn as sl
import numpy as np
from mf_wrapper import MultiflowRBFStream

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


def compute(clf_name, clf, drift, n_drifts, random_state, experiment_name, concept_kwargs):

    logging.basicConfig(filename='experiment_main.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        warnings.filterwarnings("ignore")

        if drift == 'incremental':
            incremental = True
            gradual = False
            recurring = False
        elif drift == 'gradual':
            incremental = False
            gradual = True
            recurring = False
        elif drift == 'recurring':
            incremental = False
            gradual = False
            recurring = True
        else:
            incremental = False
            gradual = False
            recurring = False

        stream_size = (concept_kwargs["n_chunks"]*concept_kwargs["chunk_size"]) / 1000
        stream_name = "stream_mf_%dd_%s_%03dk_f%02d_rs%03d" % (n_drifts, drift[0], stream_size, concept_kwargs["n_features"], random_state)

        if os.path.exists("results/raw_conf/%s/mf_%dd/%s/%s/%s.csv" % (experiment_name, n_drifts, drift, stream_name, clf_name)):
            return

        print("START: %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        logging.info("START: %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        start = time()

        stream = MultiflowRBFStream(n_chunks=concept_kwargs["n_chunks"],
                                    chunk_size=concept_kwargs["chunk_size"],
                                    n_classes=concept_kwargs["n_classes"],
                                    n_features=concept_kwargs["n_features"],
                                    n_drifts=n_drifts,
                                    incremental=incremental,
                                    gradual=gradual,
                                    recurring=recurring,
                                    random_state=random_state)

        evaluator = sl.evaluators.TestThenTrain()
        if clf.drift_detector.__class__.__name__ == "DDM":
            evaluator.process(stream, clf)
        else:
            evaluator.process(stream, clone(clf))

        filename = "results/raw_conf/%s/mf_%dd/%s/%s/%s.csv" % (experiment_name, n_drifts, drift, stream_name, clf_name)
        if not os.path.exists("results/raw_conf/%s/mf_%dd/%s/%s/" % (experiment_name, n_drifts, drift, stream_name)):
            os.makedirs("results/raw_conf/%s/mf_%dd/%s/%s/" % (experiment_name, n_drifts, drift, stream_name))
        np.savetxt(fname=filename, fmt="%d, %d, %d, %d", X=evaluator.confusion_matrix[0])

        end = time()-start

        print("DONE: %s, %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, experiment_name, end))
        logging.info("DONE - %s, %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, experiment_name, end))

    except Exception as ex:
        logging.exception("Exception in %s, %s, %s" % (drift, stream_name, clf_name))
        print("ERROR: %s, %s, %s" % (drift, stream_name, clf_name))
        traceback.print_exc()
        print(str(ex))


random_states = [0000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
drifts = ['sudden', 'incremental', 'gradual', 'recurring']
n_drifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
concept_kwargs = {
    "n_chunks": 200,
    "chunk_size": 500,
    "n_classes": 2,
    "n_features": 10,
}


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
    names = list(methods.keys())

    Parallel(n_jobs=-1)(delayed(compute)
                        (clf_name,
                         clf,
                         drift,
                         n_drifts_,
                         random_state,
                         experiment_name,
                         concept_kwargs
                         )
                        for clf_name, clf in zip(names, clfs)
                        for drift in drifts
                        for n_drifts_ in n_drifts
                        for random_state in random_states
                        )
