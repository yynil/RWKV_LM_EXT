import argparse
import os
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
import sys
import torch
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_parent_dir)
print('Add path', parent_parent_dir)
from src.model import RWKV
from src.model_ext import RwkvForSequenceEmbedding, load_ckpt_and_parse_args
from train_scripts.customer_datasets import ListDataset
from sentence_transformers.evaluation import  SimilarityFunction
import logging
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List, Literal, Optional


logger = logging.getLogger(__name__)
import torch
from torch import Tensor
from torch.utils.data import DataLoader

class EmbeddingSimilarityEvaluator:
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        data_loader : DataLoader,
        main_similarity: SimilarityFunction = None,
        name: str = "",
        write_csv: bool = True,
        precision: Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]] = None,
    ):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        :param precision: The precision to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or
            "ubinary". Defaults to None.
        """
        self.data_loader = data_loader
        self.write_csv = write_csv
        self.precision = precision


        self.main_similarity = main_similarity
        self.name = name


        self.csv_file = (
            "similarity_evaluation"
            + ("_" + name if name else "")
            + ("_" + precision if precision else "")
            + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
            "cosine_pearson",
            "cosine_spearman",
            "euclidean_pearson",
            "euclidean_spearman",
            "manhattan_pearson",
            "manhattan_spearman",
            "dot_pearson",
            "dot_spearman",
        ]


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        device = model.device
        #all matrix values
        all_pearson_cosine = []
        all_spearman_cosine = []
        all_pearson_manhattan = []
        all_spearman_manhattan = []
        all_pearson_euclidean = []
        all_spearman_euclidean = []
        all_pearson_dot = []
        all_spearman_dot = []
        for batch in self.data_loader:
            sentences1 = batch['sentence1'].to(device)
            sentences2 = batch['sentence2'].to(device)
            scores = batch['scores'].to(device)
            with torch.no_grad():
                embeddings1 = model.forward(sentences1)
                
                embeddings2 = model.forward(sentences2) 
            #convert embeddings1,embeddings2 from pytorch tensor to numpy array
            embeddings1 = embeddings1.cpu().numpy()
            embeddings2 = embeddings2.cpu().numpy()

            # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
            if self.precision == "binary":
                embeddings1 = (embeddings1 + 128).astype(np.uint8)
                embeddings2 = (embeddings2 + 128).astype(np.uint8)
            if self.precision in ("ubinary", "binary"):
                embeddings1 = np.unpackbits(embeddings1, axis=1)
                embeddings2 = np.unpackbits(embeddings2, axis=1)

            labels = scores.cpu().numpy()

            cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
            manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
            euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
            dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

            eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
            eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

            eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
            eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

            eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
            eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

            eval_pearson_dot, _ = pearsonr(labels, dot_products)
            eval_spearman_dot, _ = spearmanr(labels, dot_products)
            #append all values
            all_pearson_cosine.append(eval_pearson_cosine)
            all_spearman_cosine.append(eval_spearman_cosine)
            all_pearson_manhattan.append(eval_pearson_manhattan)
            all_spearman_manhattan.append(eval_spearman_manhattan)
            all_pearson_euclidean.append(eval_pearson_euclidean)
            all_spearman_euclidean.append(eval_spearman_euclidean)
            all_pearson_dot.append(eval_pearson_dot)
            all_spearman_dot.append(eval_spearman_dot)

            logger.info(
                "Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson_cosine, eval_spearman_cosine)
            )
            logger.info(
                "Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                    eval_pearson_manhattan, eval_spearman_manhattan
                )
            )
            logger.info(
                "Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                    eval_pearson_euclidean, eval_spearman_euclidean
                )
            )
            logger.info(
                "Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson_dot, eval_spearman_dot)
            )

        
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if os.path.exists(output_path) is False:
                os.makedirs(output_path)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        np.mean(all_pearson_cosine),
                        np.mean(all_spearman_cosine),
                        np.mean(all_pearson_euclidean),
                        np.mean(all_spearman_euclidean),
                        np.mean(all_pearson_manhattan),
                        np.mean(all_spearman_manhattan),
                        np.mean(all_pearson_dot),
                        np.mean(all_spearman_dot),
                    ]
                )


        if self.main_similarity == SimilarityFunction.COSINE:
            return np.mean(all_spearman_cosine)
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return np.mean(all_spearman_euclidean)
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return np.mean(all_spearman_manhattan)
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return np.mean(all_spearman_dot)
        elif self.main_similarity is None:
            return np.mean([all_spearman_cosine, all_spearman_manhattan, all_spearman_euclidean, all_spearman_dot])
        else:
            raise ValueError("Unknown main_similarity value")


if __name__ == '__main__':
    #enable logger to info for debugging
    logging.basicConfig(level=logging.INFO)
    sts_file = '/media/yueyulin/bigdata/data/stsbenchmark/stsbenchmark.tsv'
    import csv
    eval_data = []
    with open(sts_file, 'r',encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                eval_data.append(row)

    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer_file = os.path.join(parent_parent_dir,'tokenizer','rwkv_vocab_v20230424.txt')
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    

    # define a data collator to tokenize and pad the sentences to the same length
    pad_id =0
    embedding_id = 1
    max_seq_length = 33

    def eval_data_collator(features):
        sentence1_str = [feature['sentence1'] for feature in features]
        sentence2_str = [feature['sentence2'] for feature in features]
        scores = [float(feature['score'])/5.0 for feature in features]
        sentence1_ids = [tokenizer.encode(sentence)[0:max_seq_length-1]+[embedding_id] for sentence in sentence1_str]
        sentence2_ids = [tokenizer.encode(sentence)[0:max_seq_length-1]+[embedding_id] for sentence in sentence2_str]
        # pad the sequences to the same length
        sentence1_ids = [ids + [pad_id]*(max_seq_length-len(ids)) for ids in sentence1_ids]
        sentence2_ids = [ids + [pad_id]*(max_seq_length-len(ids)) for ids in sentence2_ids]
        return {'sentence1':torch.tensor(sentence1_ids,dtype=torch.long),'sentence2':torch.tensor(sentence2_ids,dtype=torch.long),'scores':torch.tensor(scores,dtype=torch.float)}
    
    eval_ds = ListDataset(eval_data)
    print(eval_ds[0])
    eval_dl = DataLoader(eval_ds, batch_size=16, collate_fn=eval_data_collator)
    

    base_rwkv_model = '/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    print('load base model from ',base_rwkv_model)
    rwkv_args = argparse.Namespace()
    w = load_ckpt_and_parse_args(base_rwkv_model,rwkv_args)
    print(rwkv_args)
    rwkv_args.my_pos_emb = 0
    rwkv_args.pre_ffn = 0
    rwkv_args.head_size_divisor = 8
    rwkv_args.ctx_len = 4096
    rwkv_args.dropout = 0.1
    rwkv_args.head_qk = 0
    rwkv_args.grad_cp = 0
    rwkv_args.save_per_batches = 10000
    rwkv_args.my_exit = 300
    rwkv_args.weight_decay = 0.001
    rwkv_args.lr_init = 3e-4
    rwkv_args.lr_final = 1e-5
    rwkv_args.beta1 = 0.9
    rwkv_args.beta2 = 0.99
    rwkv_args.betas = (0.9, 0.99)
    rwkv_args.layerwise_lr = 1
    rwkv_args.my_pile_stage = 1
    rwkv_args.adam_eps = 1e-8
    rwkv_args.warmup_steps = 50
    rwkv_args.tiny_att_dim = 0
    rwkv_args.model_file = base_rwkv_model
    print(rwkv_args)
    rwkv_base_model = RWKV(rwkv_args)
    print(rwkv_base_model)
    inform = rwkv_base_model.load_state_dict(w)
    print(inform)

    embedding_model = RwkvForSequenceEmbedding(rwkv_base_model)
    print(embedding_model)
    embedding_model = embedding_model.bfloat16()
    device = 'cuda'
    embedding_model.to(device)
    embedding_model.eval()
    import torch
    from torch.amp.autocast_mode import autocast
    with autocast(device_type=device,dtype=torch.bfloat16):
        evaluator = EmbeddingSimilarityEvaluator(data_loader=eval_dl,main_similarity=SimilarityFunction.COSINE,precision='float32')
        eval_value = evaluator(embedding_model,output_path='/media/yueyulin/bigdata/output/stsbenchmark',epoch=1,steps=3000)
        print(eval_value)