from river import synth
import numpy as np


class RiverRTStream:
    def __init__(self,
                 n_classes=2,
                 n_features=10,
                 n_chunks=200,
                 chunk_size=500,
                 n_drifts=1,
                 incremental=False,
                 n_centroids=10,
                 gradual=False,
                 recurring=False,
                 random_state=None
                 ):

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.n_drifts = n_drifts
        if incremental:
            self.width = chunk_size*80/n_drifts
        elif gradual:
            self.width = chunk_size*20/n_drifts
        else:
            self.width = chunk_size*1
        self.position = n_chunks*chunk_size/(n_drifts+1)
        self.n_centroids = n_centroids
        self.recurring = recurring
        self.random_state = random_state

        self.generator = self.stream(n_drifts)
        self.classes_ = range(n_classes)

    def get_chunk(self):

        if hasattr(self, "X"):
            self.previous_chunk = self.current_chunk
        else:
            data = self.generator.take(self.n_chunks*self.chunk_size)
            self.X = []
            self.y = []
            for d in data:
                self.X.append(list(d[0].values()))
                self.y.append(d[1])

            self.X = np.array(self.X)
            self.y = np.array(self.y)
            self.reset()

        self.chunk_id += 1

        if self.chunk_id < self.n_chunks:
            start, end = (
                self.chunk_size * self.chunk_id,
                self.chunk_size * self.chunk_id + self.chunk_size,
            )

            self.current_chunk = (self.X[start:end], self.y[start:end])
            return self.current_chunk
        else:
            return None

    def stream(self, idx):
        if idx > 0:
            if self.random_state:
                if self.recurring and idx < self.n_drifts / 2:
                    self.random_state -= 111
                else:
                    self.random_state += 111

            s1 = synth.RandomTree(seed_tree=self.random_state,
                                  seed_sample=self.random_state,
                                  n_classes=self.n_classes,
                                  n_num_features=self.n_features,
                                  n_cat_features=0)

            # s1 = synth.RandomRBF(seed_model=self.random_state,
            #                      seed_sample=self.random_state,
            #                      n_classes=self.n_classes,
            #                      n_features=self.n_features,
            #                      n_centroids=self.n_centroids)
            s2 = self.stream(idx-1)
            return synth.ConceptDriftStream(stream=s1,
                                            drift_stream=s2,
                                            position=self.position,
                                            width=self.width,
                                            seed=self.random_state)
        else:
            if self.random_state:
                if self.recurring:
                    self.random_state -= 111
                else:
                    self.random_state += 111
            return synth.RandomTree(seed_tree=self.random_state,
                                    seed_sample=self.random_state,
                                    n_classes=self.n_classes,
                                    n_num_features=self.n_features,
                                    n_cat_features=0)
            # return synth.RandomRBF(seed_model=self.random_state,
            #                        seed_sample=self.random_state,
            #                        n_classes=self.n_classes,
            #                        n_features=self.n_features,
            #                        n_centroids=self.n_centroids)

    def reset(self):
        self.previous_chunk = None
        self.chunk_id = -1

    def is_dry(self):
        """Checking if we have reached the end of the stream."""

        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )
