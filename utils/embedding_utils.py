# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import gc
import torch
import subprocess
import numpy as np
import typing

from functools import partial
from torch.nn.utils.rnn import pad_sequence
import torch.utils
import torch.utils.data
from tqdm import tqdm

from models.hifinn_model import HifinnLayerNormResiduePL
from utils.dataset_utils import get_embedding
from utils.file_utils import load_embeddings


def embed_queries(
        query_path: str,
        model: HifinnLayerNormResiduePL,
        input_format: str = 'all',
        device: torch.device = torch.device('cpu'),
        ids_to_keep: typing.List[str] | None = None,
        representations:str='mean_representations',
        layer: int = 33,
        padding_value: int | None = None,
        return_attention: bool = False,
        batch_size: int = 32) -> np.ndarray:
    # first, either produce or retrieve ESM embeddings
    if query_path.endswith('.fasta'):
        emb_path = esm_embed(query_path)
        emb = load_esm_embeddings_w_dl(
            emb_path,
            input_format,
            ids_to_keep,
            representations=representations,
            layer=layer,
            padding_value=padding_value,
            batch_size=batch_size)
    else:
        emb = load_esm_embeddings_w_dl(
            query_path,
            input_format,
            ids_to_keep,
            representations=representations,
            layer=layer,
            padding_value=padding_value,
            batch_size=batch_size)

    if input_format == 'sequence':
        return get_model_embeddings(emb, model, device), ['sequence']
    elif ids_to_keep is not None:
        return get_model_embeddings_from_dl(emb, model, device, return_attention=return_attention)
    else:
        return get_model_embeddings_from_dl(emb, model, device, return_attention=return_attention)
    
def get_model_embeddings(emb: torch.Tensor, model: HifinnLayerNormResiduePL, device:torch.device) -> np.ndarray:
    """
    Pass embeddings through model and return the new embeddings.

    Args:
        emb: tensor of the embeddings.
        model: Trained model which we want to pass ESM embeddings through.
        device: Device we wish to use for inference.

    Returns:
        Model embeddings, shape [batch_size, 512].
    """
    return model.forward(emb.to(device)).detach().cpu().numpy()

def get_model_embeddings_from_dl(
        dl: torch.utils.data.DataLoader, 
        model: HifinnLayerNormResiduePL, 
        device: torch.device, 
        save: bool = True, 
        bfloat16: bool = False, 
        return_attention: bool = False) -> typing.Tuple[np.ndarray, typing.List[str]]:
    """
    Pass embeddings through model and return the new embeddings.
    This kinda messes with the current set up for annotate.py
    as we load every embedding stored in ./finetuned_embeddings/
    - need to think of some better way of architecting this...

    Args:
        dl: numpy array of the embeddings.
        model: Trained model which we want to pass ESM embeddings through.
        device: Device we wish to use for inference.

    Returns:
        Model embeddings, shape [batch_size, 512].
    """
    embs = None
    ids = []
    print(f"Passing sequences through HiFiNN.")
    for batch_idx, (batch_ids, emb) in tqdm(enumerate(dl)):
        with torch.no_grad():
            # `squeeze` is so we remove the 'levels' dimension.
            if bfloat16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    #model_emb = model.forward_pass(emb.to(device)).squeeze(1).cpu().numpy()
                    if return_attention:
                        model_emb, attn = model.forward(emb.to(device))
                        model_emb, attn = model_emb.cpu().numpy(), attn.cpu().numpy()
                    else:
                        model_emb = model.forward(emb.to(device)).cpu().numpy()
            else:
                #model_emb = model.forward_pass(emb.to(device)).squeeze(1).cpu().numpy()
                if return_attention:
                    model_emb, attn = model.forward(emb.to(device))
                    model_emb, attn = model_emb.cpu().numpy(), attn.cpu().numpy()
                else:
                    model_emb = model.forward(emb.to(device)).cpu().numpy()
            if save:
                save_embeddings(model_emb, batch_ids)
                if return_attention:
                    save_embeddings(attn, [id.strip('.pt')+'_attn.pt' for id in batch_ids], dir='./attn_masks/') # add attn suffix to attention embeddings
                del model_emb
                del emb
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                if embs is None:
                    embs = model_emb
                else:
                    embs = np.append(embs, model_emb, axis=0)
                torch.cuda.empty_cache()
                #embs.append(model_emb)
                gc.collect()
                ids.extend(batch_ids)
    # if we decided to save on the fly, then load embeddings now
    if save:
        embs, ids = load_embeddings('./hifinn_embeddings/')
    print(f"Finished generating HiFiNN embeddings!")
    return embs, ids

def load_esm_embeddings_w_dl(
        query_path: str,
        input_format: str = 'all',
        ids_to_keep: typing.List[str] | None = None,
        representations: str = 'mean_representations',
        layer: int = 33,
        padding_value: int | None = None,
        batch_size: int = 32) -> torch.Tensor | torch.utils.data.DataLoader:
    if input_format == 'sequence':
         emb = torch.load(query_path + '/sequence.pt')
         return emb.unsqueeze(0)
    else:
        ds = EmbeddingDataset(
            query_path,
            ids_to_keep=ids_to_keep)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=batch_collate_fn,
            num_workers=1
        )
    return dl

def batch_collate_fn(batch: typing.List[typing.Tuple]):
    all_ids = []
    all_embs = []
    for item in batch:
        all_ids.append(item[0])
        all_embs.append(item[1])
    return all_ids, pad_sequence(all_embs, batch_first=True)

def load_esm_embeddings(
        query_path: str,
        tensor_format: str = 'all',
        ids_to_keep: typing.Optional[typing.List] = None,
        representations: str = 'mean_representations',
        layer: int = 33,
        padding_value: typing.Optional[int] = None):
    all_embs = []
    if tensor_format == 'sequence':
         emb = torch.load(query_path + '/sequence.pt')
         all_embs.append(emb.unsqueeze(0))
    else:
        if ids_to_keep is not None:
            embs_to_keep = [id + '.pt' for id in ids_to_keep]
            for emb_id in embs_to_keep:
                emb = torch.load(query_path + '/' + emb_id)
                all_embs.append(emb[representations][layer].unsqueeze(0))
        else:
            for fp in os.listdir(query_path):
                emb = torch.load(query_path + '/' + fp)
                all_embs.append(emb[representations][layer].unsqueeze(0))
    if padding_value is not None:
        # pad each embedding with the padding value
        return pad_sequence(all_embs, batch_first=True, padding_value=padding_value)
    else:
        return torch.cat(all_embs, dim=0)

def esm_embed(
        input_str:str, 
        input_format:str='fasta', 
        residue_embeddings:bool=False):
    print(f"Passing sequences through ESM. \n")
    if (input_format == 'fasta') & (residue_embeddings == False):
        cmd_array = [
        "python",
        "./esm/scripts/extract.py",
        "esm2_t33_650M_UR50D",
        input_str,
        "./esm_embeddings/",
        "--repr_layers",
        "33",
        "--include",
        "mean"
        ]
        try:
            sub_process_header = run_with_reduced_timeout(
                    cmd_array=cmd_array, timeout=None
            )
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise e

        print(f"Finished generating esm embeddings!")
        return './esm_embeddings/'
    elif (input_format == 'fasta') & (residue_embeddings == True):
        cmd_array = [
        "python",
        "./esm/scripts/extract.py",
        "esm2_t33_650M_UR50D",
        input_str,
        "./esm_embeddings/",
        "--repr_layers",
        "32",
        "--include",
        "per_tok"
        ]
        try:
            sub_process_header = run_with_reduced_timeout(
                    cmd_array=cmd_array, timeout=None
            )
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise e
        print(f"Finished generating esm embeddings!")
        return './esm_embeddings/'
    elif input_format == 'sequence':
        embed_path = embed_sequence(input_str, residue_embeddings)
        print(f"Finished generating esm embeddings!")
        return embed_path
    else:
        raise ValueError("Input format must be one of: 1) 'fasta' or 2) 'sequence'")

def run_with_reduced_timeout(cmd_array:typing.List[str], timeout:typing.Optional[int]=50):
    return subprocess.run(
        cmd_array,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=True,
    )

def embed_sequence(sequence: str, residue_embeddings:typing.Optional[bool]=False):
    import torch
    import esm

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    data = [('protein1', sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    if residue_embeddings:
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[32], return_contacts=True)
        token_representations = results["representations"][32]
    else:
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    if residue_embeddings:
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1])
    else:
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
    #with open('./esm_embeddings/sequence.pt', 'wb') as f:
    torch.save(sequence_representations[0], './esm_embeddings/sequence.pt')
    return './esm_embeddings/'

def save_embeddings(
        embeddings: torch.Tensor, 
        sequence_ids: typing.List[str], 
        dir: str='./hifinn_embeddings/'):
    """
    Assuming embeddings are stacked as 
    num_ids x embedding_dim
    """
    # create directory if it does not
    # exist already
    if not os.path.exists(dir):
        os.makedirs(dir)

    for idx, emb in enumerate(embeddings):
        torch.save(emb, dir + sequence_ids[idx])

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str, ids_to_keep:typing.List[str] | None = None):
        self.filepath = filepath
        self.ids_to_keep = ids_to_keep
        if self.ids_to_keep is None:
            self.embedding_filenames = [f for f in os.listdir(self.filepath)]
            self.ids_to_keep = [id.strip('.pt') for id in self.embedding_filenames]
        else:
            all_fnames = [f[:-3] for f in os.listdir(self.filepath)]
            id_embs = set(all_fnames).intersection(set(self.ids_to_keep))
            assert len(id_embs) == len(self.ids_to_keep), f"We only have embeddings for \
                {len(id_embs)} of these ids!\n,  \
                We do not have: {set(self.ids_to_keep).difference(set(id_embs))}"
            self.embedding_filenames = [id_emb + '.pt' for id_emb in id_embs]

    def __getitem__(self, idx: int):
        emb = get_embedding(self.filepath + '/' + self.embedding_filenames[idx])
        return self.embedding_filenames[idx], emb
    
    def __len__(self):
        return len(self.ids_to_keep)

