# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import csv
import re
import torch
import numpy as np
import typing

from models.hifinn_model import HifinnLayerNormResiduePL
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def write_to_txt(lst: typing.List[str], filename: str):
    """
    Write a list of ids to a
    text file.
    """
    with open(filename, "w") as f:
        for item in lst:
            f.write(item + "\n")
    print(f"Finished!\n")


def write_json(data: typing.Dict, filename: str):
    """
    Does what it says on the tin.
    """
    with open(filename, "w") as f:
        json.dump(data, f)


def filter_filenames(filenames: str, names_to_keep: str) -> typing.List[str]:
    """
    Filter an iterable of filenames to only
    keep those present in 'names_to_keep'.

    Assuming that files end in
    '.pt', i.e. are pytorch tensor files.
    """
    filtered_filenames = []
    for fn in filenames:
        stem = fn[:-3]  # strip '.pt. from file
        if stem in names_to_keep:
            filtered_filenames.append(fn)
    return filtered_filenames


def write_to_fasta(records: typing.List, filename: str) -> None:
    """
    Write seq record object to a fasta file.

    Args:
        records: List of seq record objects.
        filename: name of file to write.
    """
    with open(filename, "w") as f:
        SeqIO.write(records, f, "fasta")
    print(f"Finished!\n")


def read_sequences(filepath: str) -> typing.Dict[str, SeqRecord]:
    """
    Read sequences from a fasta file.
    """
    seq_records = SeqIO.parse(open(filepath), "fasta")
    id_to_record = {record.id: record for record in seq_records}
    return id_to_record


def read_txt(filename: str) -> typing.List[str]:
    """
    Read a set of ids from a text file.
    """
    ids = []
    with open(filename, "r") as f:
        for line in f:
            ids.append(line.split("\n")[0])
    return ids


def load_ids(ids_path: str) -> typing.List[str]:
    """
    Load a bunch of ids from a file.
    """
    # if the ids we want in our data set is a fasta then
    if ids_path.endswith(".fasta"):
        id_to_seq = read_sequences(ids_path)
        ids = list(id_to_seq.keys())
    elif ids_path.endswith(".txt"):
        ids = read_txt(ids_path)
    elif ids_path.endswith(".json"):
        ids = load_json(ids_path)
    else:
        raise ValueError(
            "IDs must be either a FASTA with the sequences you\
                wish to retrieve, a text file of the ids of these sequences\
                    or a json with the ids of the desired sequences stored\
                        as a list"
        )
    return ids


def check_input_format(input_str: str) -> str:
    """
    Little check to make sure our input is one of
    three format's. Not perfect, we could do another
    sense check on the contents of the folder, but
    it serves it's purpose for the most part and can
    be refactored later.
    """
    AMINO_ACIDS = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "E",
        "Q",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    if input_str.endswith(".fasta") | input_str.endswith(".faa"):
        return "fasta"
    elif os.path.exists(os.path.dirname(input_str)):
        return "folder"
    elif re.match(f"^{str(AMINO_ACIDS)}*$", input_str):
        # then is a sequence
        return "sequence"
    else:
        msg = "Input must be one of: 1) Entirely upper case sequence of \
        letters, 2) a fasta file or 3) a folder of embeddings"
        raise ValueError(msg)


def load_esm_embeddings(
    query_path: str,
    tensor_format: str = "all",
    ids_to_keep: typing.List[str] | None = None,
) -> torch.Tensor:
    """
    Load esm embeddings from a folder.
    """
    all_embs = []
    if tensor_format == "sequence":
        emb = torch.load(query_path + "/sequence.pt")
        all_embs.append(emb.unsqueeze(0))
    else:
        if ids_to_keep is not None:
            embs_to_keep = [id + ".pt" for id in ids_to_keep]
            for emb_id in embs_to_keep:
                emb = torch.load(query_path + "/" + emb_id)
                all_embs.append(emb["mean_representations"][33].unsqueeze(0))
        else:
            for fp in os.listdir(query_path):
                emb = torch.load(query_path + "/" + fp)
                all_embs.append(emb["mean_representations"][33].unsqueeze(0))
    return torch.cat(all_embs, dim=0)


def load_json(data_path: str):
    """
    Does what it says on the tin.
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def load_model(
    model_path: str,
    model_obj: HifinnLayerNormResiduePL,
    device: torch.device | str | None = None,
) -> HifinnLayerNormResiduePL:
    """
    Load weights from checkpoint into model object.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(
        model_path, weights_only=False, map_location=torch.device("cpu")  # load to cpu and copy over
    )
    # load params
    model_obj.load_state_dict(ckpt["state_dict"])
    model_obj.to(device)
    # set model to eval mode
    model_obj.eval()
    return model_obj


def load_embeddings(folder: str) -> typing.Tuple[np.ndarray, typing.List[str]]:
    """
    Load and stack embeddings from folder into numpy
    array as well as the ids of the embeddings loaded.
    """
    all_embs, all_ids = [], []
    for fp in os.listdir(folder):
        emb = torch.load(folder + "/" + fp, weights_only=False)
        all_embs.append(emb)
        all_ids.append(fp.strip(".pt"))
    return np.stack(all_embs), all_ids

def check_folder_empty(path):
    with os.scandir(path) as f_iter:
        if any(f_iter):
            return False # Folder is empty
        else:
            return True