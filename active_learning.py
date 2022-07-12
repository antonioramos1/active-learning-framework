import logging
import numpy as np
from typing import List, Callable, Tuple, Dict
from abc import ABC, abstractmethod
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from tensorflow.keras.models import Sequential


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


class ActiveLearningLabeler:
    def __init__(self, n_iters: int, samplers: List["Sampler"], init_train_size: int):
        self.n_iters = n_iters
        self.samplers = samplers
        self.init_train_size = init_train_size

    def label(
        self,
        unlabeled: np.ndarray,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        labels: np.ndarray,
        eval_function: Callable,
        model_params: Dict = None,
    ) -> List[float]:
        all_pool = unlabeled
        labels_pool = labels.copy()
        unlabeled_idx_pool = PoolData(np.array(range(len(unlabeled))))
        labeled_idx_pool = PoolData(np.array([]))

        metrics = []
        for n_iter in tqdm(range(self.n_iters)):
            if n_iter == 0:
                sampler = RandomSampler(self.init_train_size)
                idx_sample = sampler.sample_idx(unlabeled_idx_pool)
            else:
                idx_sample = np.array([]).astype(int)
                for sampler in self.samplers:
                    if (sampler.sample_size is not None) and (sampler.sample_size > len(unlabeled_idx_pool)):
                        LOGGER.info(
                            f"Not enough samples left:{len(unlabeled_idx_pool)}. Loop: {n_iter}."
                            f"Exiting Active Learning Loop."
                        )
                        return metrics
                    if not isinstance(sampler, SelfTrainingSampler) or (
                        isinstance(sampler, SelfTrainingSampler) and n_iter > sampler.skip_n_iters
                    ):
                        sampled = sampler.sample_idx(unlabeled_idx_pool, all_pool, model)
                        idx_sample = np.concatenate((idx_sample, sampled))
                        if isinstance(sampler, SelfTrainingSampler):
                            LOGGER.debug(f"Iter: {n_iter}", "Self learning total:", len(sampled))
                            pseudo_labels = sampler.pseudo_label(unlabeled_idx_pool, all_pool, model)
                            LOGGER.debug("Discrepancy pseudolabels?", labels_pool[sampled] != pseudo_labels)
                            labels_pool[sampled] = pseudo_labels

            labels_pool = self.get_labels(labels_pool)
            unlabeled_idx_pool, labeled_idx_pool = self.update_pools(unlabeled_idx_pool, labeled_idx_pool, idx_sample)
            if model_params:  # Support tf API
                model.fit(all_pool[labeled_idx_pool], labels_pool[labeled_idx_pool], **model_params)
            else:
                model.fit(all_pool[labeled_idx_pool], labels_pool[labeled_idx_pool])
            metrics.append(eval_function(model, X_test, y_test))
        return metrics

    @staticmethod
    def get_labels(data: np.ndarray) -> np.ndarray:
        """
        Dummy function to represent querying the oracle for labels.
        """
        return data

    @staticmethod
    def update_pools(
        unlabeled_pool: "PoolData", labeled_pool: "PoolData", new_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        unlabeled_pool = unlabeled_pool.remove_idx(new_idx)
        labeled_pool = labeled_pool.add(new_idx)
        LOGGER.debug(f"Unlabeled pool size:{len(unlabeled_pool)}")
        LOGGER.debug(f"Labels pool size:{len(labeled_pool)}")
        return unlabeled_pool, labeled_pool


class PoolData(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def remove_idx(self, idx):
        LOGGER.debug(f"Removed {len(idx)} instances from unlabeled data. Total unlabeled: {len(self) - len(idx)}")
        return PoolData(np.setdiff1d(self, idx))

    def add(self, new):
        LOGGER.debug(f"Added {len(new)} instances to labeled data. Total labels: {len(self) + len(new)}")
        return PoolData(np.concatenate((self, new)).astype(int))


class Sampler(ABC):
    def __init__(self, sample_size):
        self.sample_size = sample_size

    @abstractmethod
    def sample_idx(self, unlabeled_idx: np.ndarray, all_data: np.ndarray, model) -> np.ndarray:
        pass

    @staticmethod
    def predict_proba(model, data: np.ndarray, idx: np.ndarray):
        if isinstance(model, Sequential):
            return model.predict(data[idx])  # tf API
        else:
            return model.predict_proba(data[idx])  # scikit API


class RandomSampler(Sampler):
    def sample_idx(self, unlabeled_idx: np.ndarray, all_data: np.ndarray = None, model=None) -> np.ndarray:
        return np.random.choice(unlabeled_idx, size=self.sample_size, replace=False)


class UncertaintySampler(Sampler):
    def sample_idx(self, unlabeled_idx: np.ndarray, all_data: np.ndarray, model) -> np.ndarray:
        y_pred = self.predict_proba(model, all_data, unlabeled_idx)
        sorted_unlabeled_idx = self.sort_uncertainty(y_pred)[: self.sample_size]
        return unlabeled_idx[sorted_unlabeled_idx]

    @staticmethod
    def sort_uncertainty(data: np.ndarray) -> np.ndarray:
        uncertainty_scores = 1 - data.max(axis=1)
        return uncertainty_scores.argsort()[::-1]


class EntropySampler(Sampler):
    def sample_idx(self, unlabeled_idx: np.ndarray, all_data: np.ndarray, model) -> np.ndarray:
        y_pred = self.predict_proba(model, all_data, unlabeled_idx)
        sorted_unlabeled_idx = self.sort_uncertainty(y_pred)[: self.sample_size]
        return unlabeled_idx[sorted_unlabeled_idx]

    @staticmethod
    def sort_uncertainty(data: np.ndarray) -> np.ndarray:
        uncertainty_scores = entropy(data.T, base=2)
        return uncertainty_scores.argsort()[::-1]


class KMeansUncertaintySampler(UncertaintySampler):
    def __init__(self, sample_size, flatten_data=None, **kwargs):  # NOTE Only handle numericals
        super().__init__(sample_size)
        self.flatten_data = flatten_data
        self.kmeans = MiniBatchKMeans(**kwargs)

    def get_clusters(self, data: np.ndarray) -> np.ndarray:
        if self.flatten_data:
            return self.kmeans.fit_predict(data.reshape(len(data), data.shape[1] * data.shape[2]))
        else:
            return self.kmeans.fit_predict(data)

    @staticmethod
    def filter_clusters(n_cluster, clusters, data):
        return data[np.where(clusters == n_cluster)[0]]

    def sample_idx(self, unlabeled_idx: np.ndarray, all_data: np.ndarray, model) -> np.ndarray:
        clusters = self.get_clusters(all_data[unlabeled_idx])

        idx_cluster_all = np.array([]).astype(int)
        for cluster in range(self.kmeans.n_clusters):
            unlabeled_cluster_idx = self.filter_clusters(cluster, clusters, unlabeled_idx)
            y_pred = self.predict_proba(model, all_data, unlabeled_cluster_idx)

            sorted_unlabeled_idx = self.sort_uncertainty(y_pred)
            filtered_cluster_unlabeled_idx = unlabeled_cluster_idx[sorted_unlabeled_idx][: self.sample_size]
            idx_cluster_all = np.concatenate((idx_cluster_all, filtered_cluster_unlabeled_idx))
        return np.random.choice(idx_cluster_all, self.sample_size, replace=False)


class SelfTrainingSampler(Sampler):
    def __init__(self, threshold: float, skip_n_iters: int = 5, sample_size=None):
        super().__init__(sample_size)
        self.threshold = threshold
        self.skip_n_iters = skip_n_iters

    def sample_idx(self, unlabeled_idx: np.ndarray, all_data: np.ndarray, model) -> np.ndarray:
        y_pred = self.predict_proba(model, all_data, unlabeled_idx)
        return unlabeled_idx[np.any(y_pred > self.threshold, axis=1)]

    def pseudo_label(self, unlabeled_idx: np.ndarray, all_data: np.ndarray, model) -> np.ndarray:
        y_pred = self.predict_proba(model, all_data, unlabeled_idx)
        pseudo_labels = np.round(y_pred[np.any(y_pred > self.threshold, axis=1)])
        if isinstance(model, Sequential):
            return pseudo_labels
        else:
            return np.argmax(pseudo_labels, axis=1)
