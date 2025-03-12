# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import os
import typing

from collections import defaultdict
from torch.utils.data import Dataset

from utils.dataset_utils import get_embedding, convert_ec_string_to_list


class EmbeddingsAndLabelsDataset(Dataset):
    def __init__(
        self, filepath: str, ids: typing.List[str], id_to_ec, train_on_classes=False
    ) -> None:
        """
        Each label is converted to one set.
        An EC number is represented hierarchically
        as '1.2.3.4' -> ['1.', '1.2.', '1.2.3.', '1.2.3.4']
        """
        self.filepath = filepath
        self.ids = ids
        self.id_to_ec = id_to_ec
        self.train_on_classes = train_on_classes
        if self.ids is None:
            self.embedding_filenames = [f for f in os.listdir(self.filepath)]
        else:
            all_fnames = [f[:-3] for f in os.listdir(self.filepath)]
            id_embs = set(all_fnames).intersection(set(self.ids))
            assert len(id_embs) == len(
                self.ids
            ), f"We only have embeddings for \
                {len(id_embs)} of these ids!\n,  \
                We do not have: {set(self.ids).difference(set(id_embs))}"
            self.embedding_filenames = [id_emb + ".pt" for id_emb in id_embs]
        # if we want to pass over the classes
        if self.train_on_classes:
            ec_to_id = defaultdict(list)
            if self.ids is None:
                for id, ec_nums in self.id_to_ec.items():
                    for ec in ec_nums:
                        ec_to_id[ec].append(id)
            else:  # otherwise use only the ids we need
                for id in id_embs:
                    ec_nums = self.id_to_ec[id]
                    for ec in ec_nums:
                        ec_to_id[ec].append(id)
            self.ec_to_id = ec_to_id
            self.ec_nums = list(ec_to_id.keys())

    def __getitem__(self, idx):
        if self.train_on_classes:
            ec_num = self.ec_nums[idx]
            id = np.random.choice(self.ec_to_id[ec_num])
            emb = get_embedding(self.filepath + "/" + id + ".pt")
            labels = self.id_to_ec[id]
        else:
            emb = get_embedding(self.filepath + "/" + self.embedding_filenames[idx])
            labels = self.id_to_ec[self.embedding_filenames[idx].strip(".pt")]
        labels_as_list = []
        for ec in labels:
            labels_as_list.extend(convert_ec_string_to_list(ec))
        return emb, labels_as_list

    def __len__(self):
        if self.train_on_classes:
            return len(self.ec_nums)
        else:
            return len(self.embedding_filenames)
