# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import yaml
from annotators.nearest_neighbours import VectorStore
from models.hifinn_model import HifinnLayerNormResiduePL
from utils.embedding_utils import embed_queries, esm_embed
from utils.file_utils import load_ids, load_model, check_input_format


def main():
    with open("./configs/make_db.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = config["model_path"]
    index_path = config["index_path"]
    input_format = check_input_format(config["input"])

    # if we only want to build the database from a subset of
    # a folder of embeddings
    if config["ids"] is not None:
        ids = load_ids(config["ids"])
    else:
        ids = None
    device = torch.device(config['device'])
    # 1. read ESM sequences from specified folder and embed with model
    model_obj = HifinnLayerNormResiduePL(
        normalize=True,
        hidden_size_1=1024,
        output_size=512,
        per_residue_embedding=True,
        padding_value=0,
        criterion=None,
        learning_rate=None,
        weight_decay=None,
        epochs=None,
        min_lr=None,
    )
    model = load_model(model_path, model_obj, device=device)
    model.eval()
    if input_format == "fasta":
        path_to_esm_emb = esm_embed(config["input"], residue_embeddings=True)
        # 2. embed queries
        data, sorted_ids = embed_queries(
            path_to_esm_emb,
            model,
            ids_to_keep=ids,
            device=device,
            representations="representations",
            layer=32,
            padding_value=0,
        )
    elif input_format == "folder":
        data, sorted_ids = embed_queries(
            config["input"],
            model,
            device=device,
            ids_to_keep=ids,
            representations="representations",
            layer=32,
            padding_value=0,
        )
    elif input_format == "sequence":
        path_to_esm_emb = esm_embed(
            config["input_path"], "sequence", residue_embeddings=True
        )
        data, sorted_ids = embed_queries(
            path_to_esm_emb,
            model,
            device=device,
            input_format="sequence",
            representations="representations",
            layer=32,
            padding_value=0,
        )
    else:
        raise ValueError("Program should have crashed by now...")

    # 3. use sequences to generate index data structure
    db = VectorStore(
        sorted_ids,
        index_path,
        dim=512,
        metric="cosine",
        k=1,
        load_prebuilt_index=False,
        data=data,
    )

    print(f"Finished creating database!")


if __name__ == "__main__":
    main()
