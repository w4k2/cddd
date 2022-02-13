from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import numpy as np


class CLFD(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 drift_detector,
                 estimator=MLPClassifier((10, 10), random_state=123)
                 ):
        self.drift_detector = drift_detector
        self.estimator = estimator
        self.model = None
        self.model = clone(self.estimator)
        self.iterator = 0
        self.new_model = None
        self.drifts = []

    def fit(self, X, y):
        self.model = clone(self.estimator)
        self.partial_fit(X, y)

    def partial_fit(self, X, y, c):

        # Make it once per whole run
        if self.iterator == 0:
            # Init for hard reset after clone
            if self.drift_detector.__class__.__name__ == "DDM":
                self.drift_detector.__init__()

            # Store samples from each class
            [la, ia] = np.unique(y, return_index=True)
            self.stored_X = X[ia]
            self.stored_y = la

        # Reset drift
        is_drift = False

        # When detector is used
        if self.drift_detector is not None:

            # Use scikit-multiflow detector
            if hasattr(self.drift_detector, "add_element") and self.iterator > 0:
                pred_y = self.model.predict(X)

                for sample in (pred_y == y).astype(int):
                    self.drift_detector.add_element(sample)
                is_drift = self.drift_detector.detected_change()

            # Use own made detector
            elif hasattr(self.drift_detector, "partial_fit_predict"):
                is_drift = self.drift_detector.partial_fit_predict(X, y, c)

            # Use D3 detector
            elif hasattr(self.drift_detector, "add_instance"):
                for X_, y_ in zip(X, y):
                    self.drift_detector.add_instance(X_, y_)
                is_drift = self.drift_detector.detected_change()

            # When drift happens
            if is_drift:

                # Store drifted chunk index
                self.drifts.append(self.iterator)

                # Reset model
                self.model = None
                self.model = clone(self.estimator)

                # Enhance data when missing some of the classes
                if len(np.unique(y)) != len(c):
                    X = np.concatenate((X, self.stored_X))
                    y = np.concatenate((y, self.stored_y))

                # Reset for scikit-multiflow models
                if "skmultiflow" in str(self.model.__class__):
                    self.model.reset()
                    self.model.fit(X, y)

                # Reset for scikit-learn models
                else:
                    self.model.fit(X, y)

            # When drift is not detected and it is first iteration
            elif self.iterator == 0:

                # Reset for scikit-multiflow models
                if "skmultiflow" in str(self.model.__class__):
                    self.model.reset()
                    self.model.fit(X, y)

                # Reset for scikit-learn models
                else:
                    self.model.fit(X, y)

            # When drift is not detected
            else:

                # Update for scikit-multiflow models
                if "skmultiflow" in str(self.model.__class__):
                    self.model.partial_fit(X, y)

                # Update for scikit-learn models
                else:
                    self.model.partial_fit(X, y, c)

        # When detector is not used and it is first iteration
        elif self.iterator == 0:

            # Reset for scikit-multiflow models
            if "skmultiflow" in str(self.model.__class__):
                self.model.reset()
                self.model.fit(X, y)

            # Reset for scikit-learn models
            else:
                self.model.fit(X, y)

        # When detector is not used
        else:

            # Update for scikit-multiflow models
            if "skmultiflow" in str(self.model.__class__):
                self.model.partial_fit(X, y)

            # Update for scikit-learn models
            else:
                self.model.partial_fit(X, y, c)

        # Increase data chunk counter
        self.iterator += 1

        return self

    def predict(self, X):
        return self.model.predict(X)
