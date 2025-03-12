# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import typing


def jaccard_similarity(list1: list, list2: list) -> float:
    """
    Compute the Jaccard similarity between two sets.
    Takes list as input (this is a bad
    pattern and needs to be refactored).
    """
    # Convert the lists to sets
    set1 = set(list1)
    set2 = set(list2)

    # Calculate the size of the intersection and the union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Calculate and return the Jaccard similarity index
    return intersection / union


def overlap_coefficient(list1: list, list2: list) -> float:
    """
    Compute the Overlap coefficient between two sets.
    Takes list as input (this is a bad
    pattern and needs to be refactored).
    """
    # Convert the lists to sets
    set1 = set(list1)
    set2 = set(list2)

    # Calculate the size of the intersection and the union
    intersection = len(set1.intersection(set2))
    min_size = min(len(set1), len(set2))

    # Calculate and return the Jaccard similarity index
    return intersection / min_size


def pairwise_jaccard_similarity(
    list_of_lists: typing.List[typing.List[str]],
) -> typing.List[typing.List[float]]:
    """
    Compute the pairwise Jaccard similarity between
    all sets within a big list.
    """
    n = len(list_of_lists)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            matrix[i][j] = jaccard_similarity(list_of_lists[i], list_of_lists[j])
            matrix[j][i] = matrix[i][j]  # as Jaccard index is symmetric
    return matrix


def pairwise_overlap_coefficient(
    list_of_lists: typing.List[typing.List[str]],
) -> typing.List[typing.List[float]]:
    """
    Compute the pairwise overlap coefficient
    between all sets within a big list of lists.
    """
    n = len(list_of_lists)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            matrix[i][j] = overlap_coefficient(list_of_lists[i], list_of_lists[j])
            matrix[j][i] = matrix[i][j]  # as Jaccard index is symmetric
    return matrix


def pairwise_cosine_distance(X: torch.Tensor) -> torch.Tensor:
    """
    For an input of size [num_samples, embedding_size]
    compute pairwise cosine similarity of all 'num_samples'.
    We want samples which are very similar to be 'close'
    so we use 1 - cosine similarity.

    return shape: [num_samples, num_samples]
    """
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return torch.add(
        torch.mul(torch.abs(cosine(X[:, :, None], X.t()[None, :, :])), -1), 1
    )


def pairwise_cosine_similarity(X: torch.Tensor) -> torch.Tensor:
    """
    Assumes embeddings are stacked
    row-wise in tensor.
    """
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cosine(X[:, :, None], X.t()[None, :, :])


def pairwise_euclidean(X: torch.Tensor) -> torch.Tensor:
    """
    Assumes embeddings are stacked
    row-wise in tensor.
    """
    dist = torch.norm(X[:, None] - X, dim=2, p=2)
    return dist


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Return the pearson correlation coefficient
    between two tensors.
    """
    centred_x = x - torch.mean(x)
    centred_y = y - torch.mean(y)
    return torch.sum(centred_x * centred_y) / (
        torch.sqrt(torch.sum(centred_x**2)) * torch.sqrt(torch.sum(centred_y**2))
    )
