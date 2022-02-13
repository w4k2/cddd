from tqdm import tqdm
import numpy as np
from scipy.stats import logistic
import os


def accuracy(tn, fp, fn, tp):
    return np.nan_to_num((tp+tn)/(tn+fp+fn+tp))


def recall(tn, fp, fn, tp):
    return np.nan_to_num(tp/(tp+fn))


def specificity(tn, fp, fn, tp):
    return np.nan_to_num(tn/(tn+fp))


def precision(tn, fp, fn, tp):
    return np.nan_to_num(tp/(tp+fp))


def f1_score(tn, fp, fn, tp):
    prc = precision(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num(2*(prc*rec)/(prc+rec))


def balanced_accuracy(tn, fp, fn, tp):
    spc = specificity(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num((1/2)*(spc+rec))


def g_mean(tn, fp, fn, tp):
    spc = specificity(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num(np.sqrt(spc*rec))


def mcc(tn, fp, fn, tp):
    return np.nan_to_num(((tp*tn)-(fp*fn))/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))


def performance_loss(tn, fp, fn, tp, drifts, space):
    scores = accuracy(tn, fp, fn, tp)
    # drifts = [34, 66, 100, 132, 166]

    medians = []
    mins = []
    start = 0
    for drift_index in drifts:
        medians.append(np.median(scores[start:drift_index]))
        mins.append(np.min(scores[drift_index:drift_index+space]))
        start = drift_index
    medians.append(np.median(scores[start:]))

    performance_loss = []
    for i, min in enumerate(mins):
        min_median = np.min(medians[i:i+2])
        if min_median == 0:
            performance_loss.append(1)
        elif(min_median < min):
            performance_loss.append(0)
        else:
            performance_loss.append((min_median - min) / min_median)

    return np.array(performance_loss)


def recovery(tn, fp, fn, tp, drifts, space):
    scores = accuracy(tn, fp, fn, tp)
    length = len(scores)

    medians = []
    mins = []
    start = 0
    for drift_index in drifts:
        medians.append(np.median(scores[start:drift_index]))
        mins.append(np.min(scores[drift_index-5:drift_index+space]))
        start = drift_index
    medians.append(np.median(scores[start:]))

    recovery_lengths = []
    for index, drift_index in enumerate(drifts):
        score_index = drift_index+1
        while scores[score_index] < 0.95*medians[index+1]:
            score_index += 1
        recovery_lengths.append((score_index-drift_index)/length)

    return recovery_lengths


def drift_medians(tn, fp, fn, tp, drifts):
    scores = g_mean(tn, fp, fn, tp)

    medians = []
    start = 0
    for drift_index in drifts:
        medians.append(np.median(scores[start:drift_index]))
        start = drift_index
    medians.append(np.median(scores[start:]))

    return np.array(medians)


def drift_means(tn, fp, fn, tp, drifts):
    scores = g_mean(tn, fp, fn, tp)

    medians = []
    start = 0
    for drift_index in drifts:
        medians.append(np.mean(scores[start:drift_index]))
        start = drift_index
    medians.append(np.mean(scores[start:]))

    return np.array(medians)


def _sigmoid(n_chunks, chunk_size, sigmoid_spacing, n_drifts):
    n_samples = n_chunks * chunk_size
    period = (
        int((n_samples) / (n_drifts)) if n_drifts > 0 else int(n_samples)
    )
    css = sigmoid_spacing if sigmoid_spacing is not None else 9999
    _probabilities = (
        logistic.cdf(
            np.concatenate(
                [
                    np.linspace(
                        -css if i % 2 else css, css if i % 2 else -css, period
                    )
                    for i in range(n_drifts)
                ]
            )
        )
        if n_drifts > 0
        else np.ones(n_samples)
    )

    # Ensuring that the probabilities are divisible by the number of drifts
    probabilities = np.ones(n_chunks * chunk_size) * _probabilities[-1]
    probabilities[: _probabilities.shape[0]] = _probabilities

    return (period, probabilities)


def calculate_metrics(methods, streams, metrics, experiment_name, recount=False):
    data = {}
    metrics_func = {
                    "accuracy": accuracy,
                    "recall": recall,
                    "specificity": specificity,
                    "precision": precision,
                    "f1_score": f1_score,
                    "balanced_accuracy": balanced_accuracy,
                    "g_mean": g_mean,
                    "mcc": mcc,
                    "performance_loss": performance_loss,
                    "recovery": recovery,
                    "drift_medians": drift_medians,
                    "drift_means": drift_means,

    }

    for stream_name in streams:
        for clf_name in methods:
            try:
                filename = "results/raw_conf/%s/%s/%s.csv" % (experiment_name, stream_name, clf_name)
                data[stream_name, clf_name] = np.genfromtxt(filename, delimiter=',', dtype=np.int16)
            except Exception:
                data[stream_name, clf_name] = None
                print("Error in loading data", "results/raw_conf/%s/%s/%s.csv" % (experiment_name, stream_name, clf_name), clf_name)

    drift_metrics = ["performance_loss", "recovery", "drift_medians", "drift_means"]
    for stream_name in tqdm(streams, "Metrics %s" % experiment_name):
        if any(dm in metrics for dm in drift_metrics):
            generator = stream_name.split("_")[2]
            n_drifts = int(stream_name.split("_")[3][0])
            drift_type = stream_name.split("_")[4]
            chunk_size = 500
            n_chunks = int(int(stream_name.split("_")[5][:-1])*1000/chunk_size)

        for clf_name in methods:
            if data[stream_name, clf_name] is None:
                continue
            for metric in metrics:
                if os.path.exists("results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)) and not recount:
                    continue

                tn = data[stream_name, clf_name][:, 0]
                fp = data[stream_name, clf_name][:, 1]
                fn = data[stream_name, clf_name][:, 2]
                tp = data[stream_name, clf_name][:, 3]

                if metric in drift_metrics:
                    if generator == "sl":
                        if drift_type == "s":
                            dc_proba = _sigmoid(n_chunks, chunk_size, None, n_drifts)[1]
                            drifts = np.unique(np.rint((np.array(np.where(np.abs(dc_proba-0.5) < 0.3)[0])/chunk_size)).astype(int))
                        else:
                            dc_proba = _sigmoid(n_chunks, chunk_size, 5, n_drifts)[1]
                            drifts = np.unique(np.rint((np.array(np.where(np.abs(dc_proba-0.5) < 0.0001)[0])/chunk_size)).astype(int))

                    elif generator == "mf":
                        if n_drifts == 1:
                            drifts = [int(n_chunks / 2)]
                        else:
                            dr = n_chunks / n_drifts
                            drifts = np.arange(dr, n_chunks, dr, dtype=int)

                    # print(drifts, stream_name)
                    space = n_chunks-drifts[-1]-5
                    # print(space)
                    result = metrics_func[metric](tn, fp, fn, tp, drifts, space)
                else:
                    result = metrics_func[metric](tn, fp, fn, tp)

                filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)

                if not os.path.exists("results/raw_metrics/%s/%s/%s/" % (experiment_name, stream_name, metric)):
                    os.makedirs("results/raw_metrics/%s/%s/%s/" % (experiment_name, stream_name, metric))

                np.savetxt(fname=filename, fmt="%f", X=result)
