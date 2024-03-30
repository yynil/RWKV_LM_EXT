import argparse
import sys
import os
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_parent_dir)
print('Add path', parent_parent_dir)
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
from src.model import RWKV
from src.model_ext import load_embedding_ckpt_and_parse_args,RwkvForSequenceEmbedding
import gzip
import torch
import torch.amp as amp
import numpy as np
from sklearn.metrics import ndcg_score,average_precision_score
from sentence_transformers.util import cos_sim
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_rwkv_model', type=str, default='/media/yueyulin/bigdata/output/askubuntu/trainable_model/epoch_1_step_3000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth', help='base rwkv model for training')
    parser.add_argument('--data_path', type=str, default='/media/yueyulin/bigdata/data/askubuntu', help='Path to the training data')
    parser.add_argument('--max_seq_length', type=int, default=32, help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=0.05, help='Temperature parameter for the softmax')
    args = parser.parse_args()

    print('load base model from ',args.base_rwkv_model)
    rwkv_args = argparse.Namespace()
    w = load_embedding_ckpt_and_parse_args(args.base_rwkv_model, rwkv_args)
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
    rwkv_args.model_file = args.base_rwkv_model
    rwkv_base_model = RWKV(rwkv_args)
    print(rwkv_base_model)
    inform = rwkv_base_model.load_state_dict(w,strict=False)
    print(inform)
    askubuntu_folder = args.data_path
    embedding_model = RwkvForSequenceEmbedding(rwkv_base_model)
    print(embedding_model)
    embedding_model = embedding_model.bfloat16()
    embedding_model.eval()
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer_file = os.path.join(parent_parent_dir,'tokenizer','rwkv_vocab_v20230424.txt')
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    
    # Read the corpus
    corpus = {}
    dev_test_ids = set()
    with gzip.open(os.path.join(askubuntu_folder, "text_tokenized.txt.gz"), "rt", encoding="utf8") as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            id = splits[0]
            title = splits[1]
            corpus[id] = title

    # Read dev & test dataset
    def read_eval_dataset(filepath):
        dataset = []
        with open(filepath) as fIn:
            for line in fIn:
                query_id, relevant_id, candidate_ids, bm25_scores = line.strip().split("\t")
                if len(relevant_id) == 0:  # Skip examples without relevant entries
                    continue

                relevant_id = relevant_id.split(" ")
                candidate_ids = candidate_ids.split(" ")
                negative_ids = set(candidate_ids) - set(relevant_id)
                dataset.append(
                    {
                        "query": corpus[query_id],
                        "positive": [corpus[pid] for pid in relevant_id],
                        "negative": [corpus[pid] for pid in negative_ids],
                    }
                )
                dev_test_ids.add(query_id)
                dev_test_ids.update(candidate_ids)
        return dataset
    

    dev_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "dev.txt"))
    test_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "test.txt"))
    print(dev_dataset[0])
    print(test_dataset[0])

    def calculate_embeddings(input_ids, model):
        with torch.no_grad():
            outputs = model(input_ids)
        return outputs

    # Evaluate the model
    def evalute(query,positive,negative,tokenizer,model,
                all_mrr_scores = [],
                all_ndcg_scores = [],
                all_ap_scores = []):
        if len(positive) == 0 or len(negative) == 0:
            return
        query_ids = tokenizer.encode(query)+[model.embedding_id]
        positive_ids = [tokenizer.encode(p)+[model.embedding_id] for p in positive]
        negative_ids = [tokenizer.encode(n)+[model.embedding_id] for n in negative]
        query_embedding = calculate_embeddings(torch.tensor(query_ids,device='cuda',dtype=torch.long).unsqueeze(0), model)
        docs = positive_ids + negative_ids
        max_len_doc = max([len(doc) for doc in docs])
        docs = [doc + [0]*(max_len_doc-len(doc)) for doc in docs]
        docs = torch.tensor(docs,device='cuda',dtype=torch.long)
        doc_embeddings = calculate_embeddings(docs, model)
        is_relevant = [1]*len(positive_ids) + [0]*len(negative_ids) 
        pred_scores = cos_sim(query_embedding, doc_embeddings).squeeze(0)
        pred_scores_argsort = torch.argsort(-pred_scores)  # Sort in decreasing order
        pred_scores = pred_scores.cpu().tolist()
        mrr_score = 0
        at_k = 10
        
        for rank, index in enumerate(pred_scores_argsort[0 : at_k]):
            if is_relevant[index]:
                mrr_score = 1 / (rank + 1)
                break
        all_mrr_scores.append(mrr_score)

        # Compute NDCG score
        all_ndcg_scores.append(ndcg_score([is_relevant], [pred_scores], k=at_k))
        # Compute AP
        all_ap_scores.append(average_precision_score(is_relevant, pred_scores))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)

        return {"map": mean_ap, "mrr": mean_mrr, "ndcg": mean_ndcg}
    query = test_dataset[0]["query"]
    positive = test_dataset[0]["positive"]
    negative = test_dataset[0]["negative"]
    embedding_model = embedding_model.to('cuda')
    with amp.autocast_mode.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
        results = evalute(query,positive,negative,tokenizer,embedding_model)
    print(results)

    with amp.autocast_mode.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []
        for data in test_dataset:
            query = data["query"]
            positive = data["positive"]
            negative = data["negative"]
            evalute(query, positive, negative, tokenizer, embedding_model,all_mrr_scores, all_ndcg_scores, all_ap_scores)
        for data in dev_dataset:
            query = data["query"]
            positive = data["positive"]
            negative = data["negative"]
            evalute(query, positive, negative, tokenizer, embedding_model,all_mrr_scores, all_ndcg_scores, all_ap_scores)

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)
        print(f"Mean AP: {mean_ap}")
        print(f"Mean MRR: {mean_mrr}")
        print(f"Mean NDCG: {mean_ndcg}")
