# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

from utils.distances import (
    pairwise_cosine_similarity,
    pairwise_jaccard_similarity,
    pairwise_overlap_coefficient,
)


class JaccardLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, embeddings, labels):
        # compute the pairwise jaccard similarity between all the labels in a batch
        jaccs = torch.tensor(
            pairwise_jaccard_similarity(labels), device=embeddings.device
        )
        cosines = torch.abs(pairwise_cosine_similarity(embeddings))
        lower_indices = torch.tril_indices(
            cosines.shape[0], cosines.shape[1], offset=-1
        )
        lower_jaccs = jaccs[lower_indices].reshape(-1, 1)
        lower_cosines = torch.clip(cosines[lower_indices].reshape(-1, 1), 0, 1)
        return torch.nn.functional.mse_loss(lower_cosines, lower_jaccs)


class OverlapLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, embeddings, labels):
        overlap = torch.tensor(
            pairwise_overlap_coefficient(labels),
            device=embeddings.device,
            dtype=embeddings.dtype,
        )  # compute the pairwise overlap coefficient between all the labels in a batch
        cosines = pairwise_cosine_similarity(embeddings)
        lower_indices = torch.tril_indices(
            cosines.shape[0], cosines.shape[1], offset=-1
        )
        lower_overlap = overlap[lower_indices].reshape(-1, 1)
        lower_cosines = torch.clip(cosines[lower_indices].reshape(-1, 1), 0, 1)
        return torch.nn.functional.mse_loss(lower_cosines, lower_overlap)
