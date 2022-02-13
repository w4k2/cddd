import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import median_filter
from math import e


class CentroidDistanceDriftDetectorV2:

    def __init__(self, sensitive=0.2, distance_p=2, filter_size=(3)):

        self.distance_p = distance_p
        self.filter_size = filter_size
        self.sensitive = sensitive

        self.iterator = 0
        self.cd_idx = 0
        self.concepts = []
        self.centroids = None
        self.con_array = []
        self.warning = False

    def partial_fit_predict(self, X, y, c):

        # If there are no centroids
        if self.centroids is None:
            self.centroids = [np.mean(X, axis=0).tolist()]
            self.iterator += 1
            return False

        # Save centroids
        self.centroids.append(np.mean(X, axis=0).tolist())

        # Calculate actual distances
        distances = cdist(self.centroids[self.cd_idx:self.iterator], self.centroids[self.cd_idx:self.iterator], "cityblock")
        distances = median_filter(distances, size=(self.filter_size))

        # Calculate previous distances
        dist = cdist([self.centroids[self.iterator]], self.centroids[self.cd_idx:self.iterator], "cityblock")[0]
        dist = median_filter(dist, size=(self.filter_size))

        # Check if it second iteration
        if self.iterator == 1:
            # Store 0
            self.mean_distances = [0]

        # Calculate mean of distances
        mn = np.mean(dist)

        # Calculate alpha
        x = self.iterator-self.cd_idx-10
        alpha = 1.5+(2/(1+e**(x*0.5)))

        # Calculate threshold condition
        con = np.mean(self.mean_distances[self.cd_idx:self.iterator]) + alpha*np.std(self.mean_distances[self.cd_idx:self.iterator])

        # Store mean and condition value for visualization
        self.mean_distances.append(mn)
        self.con_array.append(con)

        if self.warning and mn > self.con and self.iterator > 3:
            # Drift detected
            self.warning = False
            self.concepts.append(self.iterator)
            self.cd_idx = self.iterator
            self.iterator += 1
            return True

        if mn > con:
            # Warning status
            self.warning = True
            self.con = con
        else:
            # Remove warning status
            self.warning = False

        self.iterator += 1
        return False
