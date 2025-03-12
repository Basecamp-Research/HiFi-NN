# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import yaml
from models.hifinn_model import HifinnLayerNormResiduePL
from utils.file_utils import check_input_format, load_model, load_json, write_json, check_folder_empty
from utils.embedding_utils import esm_embed, embed_queries
from annotators.nearest_neighbours import query_database, filter_by_distance


def main():
    with open("./configs/annotate.yaml", "r") as f:
        config = yaml.safe_load(f) 
    # check that the hifinn_embeddings folder is empty
    assert check_folder_empty('./hifinn_embeddings/'), './hifinn_embeddings/ must be empty before we embed the sequences!'
    
    device = torch.device(config['device'])  # device to load the model onto
    input_format = check_input_format(
        config["input"]
    )  # 'fasta', 'folder' or 'sequence'

    # 1. load requisite files
    # Create per residue model
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
    # load model weights
    model = load_model(config["model_path"], model_obj, device=device)
    model.eval()  # set model to eval mode for inference
    id_to_annotation = load_json(
        config["id_to_annotation"]
    )  # load id to annotation mapping
    ids = load_json(config["ids"])  # load the ids the vectors are indexed with
    index = config["index"]  # store path to index in variable, is this even necessary?

    if input_format == "fasta":
        path_to_esm_emb = esm_embed(
            config["input"], residue_embeddings=True
        )  # embed the sequences with ESM-2
        # 2. embed queries with our model
        queries, query_ids = embed_queries(
            path_to_esm_emb,
            model,
            representations="representations",  # key used to extract embeddings from ESM-2 output
            layer=32,  # the layer we extracted the embeddings from
            padding_value=0,  # the value we wish to pad our sequences with
            device=device,
        )
    elif input_format == "folder":
        queries, query_ids = embed_queries(
            config["input"],
            model,
            representations="representations",
            layer=32,
            padding_value=0,
            device=device,
        )
    elif input_format == "sequence":
        path_to_esm_emb = esm_embed(
            config["input"], "sequence", residue_embeddings=True
        )
        queries, query_ids = embed_queries(
            path_to_esm_emb,
            model,
            input_format="sequence",
            representations="representations",
            layer=32,
            padding_value=0,
            device=device,
        )
    else:
        raise ValueError("Program should have crashed by now...")
    # free up memory
    del model
    
    # 3. query database
    id_to_neighbours = query_database(
        queries,
        query_ids,
        ids,
        index,
        id_to_annotation,
        config["k"],
        return_distance=config["return_distance"],
        return_confidence=config["return_confidence"],
        dim=512,
        metric="cosine",
        gpu=False,
    )
    id_to_neighbours = filter_by_distance(
        id_to_neighbours, dist_cutoff=config["distance_cutoff"]
    )

    if config["output_path"] is None:
        if input_format == "sequence":
            write_json(id_to_neighbours, f"./sequence_k{config['k']}_annotations.json")
        elif input_format == "fasta":
            print(
                f"Saving annotations to: ./{str(config['input']).split('/')[-1].strip('.fasta')}_k{config['k']}_annotations.json"
            )
            write_json(
                id_to_neighbours,
                f"./{str(config['input']).split('/')[-1].strip('.fasta')}_k{config['k']}_annotations.json",
            )
        else:
            print(
                f"Saving annotations to: ./{str(config['input']).split('/')[-2].strip('.fasta')}_k{config['k']}_annotations.json"
            )
            write_json(
                id_to_neighbours,
                f"./{str(config['input']).split('/')[-2].strip('.fasta')}_k{config['k']}_annotations.json",
            )
    else:
        write_json(id_to_neighbours, config["output_path"])


if __name__ == "__main__":
    main()
