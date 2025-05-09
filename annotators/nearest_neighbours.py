# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import faiss
import numpy as np
import typing
import os
from collections import defaultdict


def filter_by_distance(
    annos: typing.Dict[str, typing.Dict[str, typing.Tuple[float, float]]],
    dist_cutoff: float,
) -> typing.Dict[str, typing.Dict[str, float]]:
    """
    Filter predictions by a distance threshold.

    Args:
        annos: The predictions
            of our method. Dictionary mapping
            id to a list of confidence and distance.

        dist_cutoff: The distance cutoff we
            we wish to use to filter our predictions.

    Returns:
        The filtered annos with the
        confidence score retained.
    """
    if dist_cutoff < 0:
        raise ValueError(f"Distance cutoff cannot be negative. Distance cutoff entered: {dist_cutoff}")
    new_annos: defaultdict = defaultdict(dict)
    for id, preds in annos.items():
        for ec, scores in preds.items():
            if scores[1] < dist_cutoff:
                new_annos[id].update({ec: {
                    'conf':scores[0],
                    'dist':scores[1]}
                })
    return new_annos


def query_database(
    queries: np.ndarray,
    query_ids: typing.List[str],
    sorted_ids: typing.List[str],
    index_path: str,
    id_to_ec: typing.Dict[str, typing.List[str]],
    k: int = 10,
    return_distance: bool = False,
    return_confidence: bool = True,
    dim: int = 512,
    gpu: bool = False,
    metric: str = "L2",
    T: float = 0.001,
) -> typing.Dict[str, typing.Dict[str, float]]:
    """
    Search queries against FAISS vector index.
    Return either the confidence or the distance or
    both.

    Args:
        queries: N x D numpy array of vectors to
            search for hits to.
        query_ids: the N ids of the queries
            this is so we can map the hits back to the
            query id.
        sorted_ids: list of ids where the
            position maps to the row index in the vector
            index structure.
        index_path: path to FAISS vector index.
        k: the number of neighbours to retrieve.
        return_distance: indicator if we should
            return the distance to the closest hit for
            a given EC.
        return_confidence: indicator if we should
            return the confidence in an annotation we
            have predicted.
        id_to_ec: mapping of id to
            the associated ec number for each id in our
            vector index. Used for annotating the queries
            by transferring the annotations of their hits.
        dim: dimension of the vectors indexed.
        gpu: To use the GPU for searching or not.
            Recommended for a large index structure or a large
            number of sequences.
        metric: Either L2, indicating Euclidean distance,
            or 'cosine' indicating cosine similarity (not
            technically a metric, I know I know).

    Returns:
        the id of each query mapped to the
        annotations it has been given alongside either
        the minimum distance to the hit or the confidence
        associated with the annotation.
    """
    # Check input args
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    # create our vector store object
    db = VectorStore(
        sorted_ids,
        index_path=index_path,
        k=k,
        load_prebuilt_index=True,
        dim=dim,
        metric=metric,
    )
    dist, indices = db.query(queries, gpu=gpu)  # query the database
    # if metric is cosine similarity, we want to return a distance
    # so we take 1-cosine similarity
    if metric == "cosine":
        for i in range(len(dist)):
            dist[i] = [1 - d for d in dist[i]]  # update sims to be distances

    if (return_distance is True) and (return_confidence is False):
        preds = db.annotate_w_ec_nums(indices, id_to_ec)
        return get_label_distances(dist, preds, query_ids)
    elif (return_distance is False) and (return_confidence is True):
        preds = db.annotate_w_ec_nums(indices, id_to_ec)
        return get_confidence_scores(dist, preds, query_ids, T=T)
    elif return_distance and return_confidence:
        preds = db.annotate_w_ec_nums(indices, id_to_ec)
        return get_confidence_and_distance(dist, preds, query_ids, T=T)
    else:  # otherwise, just return the mapping of query to ids in vector store.
        return db.annotate(indices, query_ids)


def get_confidence_scores(
    distances: typing.List[typing.List[float]],
    preds: typing.List[typing.List[str]],
    ids: typing.List[str],
    T: float = 0.001,
) -> typing.Dict[str, typing.Dict[str, float]]:
    """
    Given some distances to hits and the associated id of
    the closest hit we return the confidence in the prediction.

    Args:
        distances: a list of distances
            associated with each query.
        preds: the list of labels associated
            with each hit, for each query. Of size n x k where
            n is the number of queries and k is the number of hits.
            `len(preds[i][j])` is the number of EC numbers which
            neighbour j for query i has.
        ids: the ids of the query, so we can map to
            retrieved labels.
        T: scaling parameter.

    Returns:
        Mapping of query id to a mapping of
        EC number to confidence score.
    """
    id_to_preds = {}
    for i in range(len(distances)):
        denominator = np.sum([np.exp((-d) / T) for d in distances[i]])
        label_to_prob: defaultdict = defaultdict(np.float64)
        for j, labelset in enumerate(preds[i]):
            for label in labelset:
                label_to_prob[label] += np.exp(-distances[i][j] / T) / denominator
        id_to_preds[ids[i]] = label_to_prob
    return id_to_preds


def get_label_distances(
    distances: typing.List[typing.List[float]],
    preds: typing.List[typing.List[str]],
    ids: typing.List[str],
) -> typing.Dict[str, typing.Dict[str, float]]:
    """
    Given some distances to hits and the associated id of
    the closest hit we return the confidence in the prediction.

    Args:
        distances: a list of distances
            associated with each query.
        preds: the list of labels associated
            with each hit, for each query. Of size n x k where
            n is the number of queries and k is the number of hits.
            `len(preds[i][j])` is the number of EC numbers which
            neighbour j for query i has.
        ids: the ids of the query, so we can map to
            retrieved labels.

    Returns:
        Mapping of query id to a mapping of
        EC number to the minimum distance of the query to
        the neighbour which contained this EC number.
    """
    id_to_preds = {}
    for i in range(len(distances)):
        label_to_dist = defaultdict(np.float64)
        for j, labelset in enumerate(preds[i]):
            for label in labelset:
                # here, just take minimum distance to label
                label_to_dist[label] = min(
                    float(label_to_dist.get(label, 1e9)), float(distances[i][j])
                )
        id_to_preds[ids[i]] = label_to_dist
    return id_to_preds


def get_confidence_and_distance(
    distances: typing.List[typing.List[float]],
    preds: typing.List[typing.List[str]],
    ids: typing.List[str],
    T: float = 0.001,
) -> typing.Dict[str, typing.Dict[str, float]]:
    """
    Given some distances to hits and the associated id of
    the closest hit we return the confidence in the prediction.

    Args:
        distances: a list of distances
            associated with each query.
        preds: the list of labels associated
            with each hit, for each query. Of size n x k where
            n is the number of queries and k is the number of hits.
            `len(preds[i][j])` is the number of EC numbers which
            neighbour j for query i has.
        ids: the ids of the query, so we can map to
            retrieved labels.
        T: scaling parameter.

    Returns:
        Mapping of query id to a mapping of
        EC number to a list with [confidence score, distance].
    """
    id_to_preds = {}
    for i in range(len(distances)):
        denominator = np.sum([np.exp((-d) / T) for d in distances[i]])
        label_to_prob = defaultdict(np.float64)
        label_to_dist = defaultdict(np.float64)
        for j, labelset in enumerate(preds[i]):
            for label in labelset:
                # here, just take minimum distance to label
                label_to_dist[label] = min(
                    float(label_to_dist.get(label, 1e9)), float(distances[i][j])
                )
                label_to_prob[label] += np.exp(-distances[i][j] / T) / denominator
        id_to_preds[ids[i]] = {
            label: (label_to_prob[label], label_to_dist[label])
            for label in label_to_dist.keys()
        }
    return id_to_preds


class VectorStore:
    def __init__(
        self,
        sorted_ids: typing.List[str],
        index_path: str,
        dim: int = 512,
        metric: str = "L2",
        k: int = 1,
        load_prebuilt_index: bool = False,
        data: np.ndarray | None = None,
    ):
        """
        Class to handle the vector index structure.

        Args:
            sorted_ids: list of ids where the index matches
                the row index of the vector database.
            index_path: the path to the index data structure.
            dim: the dimension of the vectors in the data base.
            metric: Either L2, indicating Euclidean distance,
                or 'cosine' indicating cosine similarity (not
                technically a metric, I know I know).
            k: the number of neighbours to retrieve.
            load_prebuilt_index: whether to load a pre-built
                index to the vector data base or not. If true, uses the
                index path to load the index.
            data: data to construct index from if prebuilt
                index is not loaded.
        """
        self.ids = sorted_ids
        self.metric = metric
        if (self.metric == "cosine") & (data is not None):
            faiss.normalize_L2(data)
        if load_prebuilt_index:
            self.read_index(index_path)
        else:
            self._build(data, dim, index_path)
        self.k = k

    def _build(self, data: np.ndarray, dim: int, index_path: str):
        """
        Build database index structure.

        Args:
            data: N x D array of data to be indexed.
            dim: the dimension of the vectors.
            index_path: path to save the index to.
        """
        if self.metric == "cosine":
            self.index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(data)
        self.save_index(index_path + ".index")
        print(f"Index saved to: {index_path+'.index'}\n")
        # save ids which reflect order of index which has been built
        with open(f"{index_path.strip('.index')}_sorted_ids.json", "w") as f:
            json.dump(self.ids, f)

    def query(
        self, queries: np.ndarray, gpu: bool = False
    ) -> typing.Tuple[np.ndarray, typing.List[typing.List[int]]]:
        """
        Search our queries against the vector
        database.

        Args:
            queries: the query vectors to search for
                hits with.
            gpu: whether to use the GPU or not.

        Returns:
            Distance to each hit
            and the indices of them in the vector index.
        """
        if gpu:
            # move index to gpu
            print(f"Moving index to GPU!\n")
            resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(resources, 5, self.index)
        if self.metric == "cosine":
            faiss.normalize_L2(queries)
        return self.index.search(queries, self.k)

    def annotate(
        self, indices: typing.List[typing.List[int]], ids: typing.List[str]
    ) -> typing.Dict[str, typing.List[str]]:
        """
        Go from indices (list of lists)
        to the ids of the hits they represent.

        Args:
            indices: the indices of the
            hits for each query.
            ids:  the ids of each entry in our
                vector structure.

        Returns:
            Map of query to the ids of its
            hits.
        """
        preds = {}
        for row in range(len(indices)):
            # index into the id of neighbours in training set to query
            neighbours = [self.ids[i] for i in indices[row, :]]
            preds[ids[row]] = neighbours
        return preds

    def read_index(self, index_path: str) -> None:
        """
        Read index structure from a specified filepath.

        Args:
            index_path: path to the index.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")
        self.index = faiss.read_index(index_path)

    def save_index(self, index_dir: str) -> None:
        """
        Save the index structure to a specified filepath

        Args:
            index_dir: path to save the index to.
        """
        faiss.write_index(self.index, index_dir)

    def annotate_w_ec_nums(
        self,
        indices: typing.List[typing.List[int]],
        id_to_ec: typing.Dict[str, typing.List[str]],
    ) -> typing.List[typing.List[str]]:
        """
        Want to associate each annotation with
        the neighbour it came from.
        We annotate each query with a list of lists
        essentially this will be 'k' lists each with a set
        of EC numbers.

        Args:
            indices: the indices of the hits
                for each query.
            id_to_ec: the mapping of id to ec,
                to be used when transferring the annotations
                of the hits to the query.

        Returns:
            k lists with a list of the EC numbers
            associated with each hit for a given query.
        """
        preds = []
        for row in range(len(indices)):
            # index into the id of neighbours in training set to query
            # here, id_to_ec should be the mapping for the ground truth labels.
            neighbours = [self.ids[i] for i in indices[row, :]]
            if isinstance(neighbours[0], list):
                pred = [id_to_ec[neighbour[0].strip(".pt")] for neighbour in neighbours]
            else:
                pred = [id_to_ec[neighbour.strip(".pt")] for neighbour in neighbours]
            preds.append(pred)
        return preds
