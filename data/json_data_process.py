import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = None
from torch.utils.data import DataLoader
import random
import torch
from functools import partial
import math

def tokenization_function(examples):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
    query_ids = []
    for query in examples['query']:
        query_ids.append(tokenizer.encode(query, add_special_tokens=False))
    pos_ids = []
    for pos in examples['pos']:
        p_ids = []
        for p in pos:
            p_ids.append(tokenizer.encode(p, add_special_tokens=False))
        pos_ids.append(p_ids)
    neg_ids = []
    for neg in examples['neg']:
        n_ids = []
        for n in neg:
            n_ids.append(tokenizer.encode(n, add_special_tokens=False))
        neg_ids.append(n_ids)
    return {'query_ids': query_ids, 'pos_ids': pos_ids, 'neg_ids': neg_ids}
def cross_collate_fn(examples, cls_id, max_len,sep_id, pad_id,train_group_size=8):
    input_ids = []
    for example in examples:
        query_id_choice = example['query_ids']
        if len(query_id_choice) >= max_len-22:
            #we ask for at least 20 tokens for pos and neg
            query_id_choice = query_id_choice[:max_len-22]
        query_id_choice = query_id_choice + [cls_id]
         # 随机选择pos_ids中的一个元素
        pos_choice = random.choice(example['pos_ids'])
        input_ids_pos = query_id_choice + [sep_id] + pos_choice
        if len(input_ids_pos) >= max_len-1:
            input_ids_pos = input_ids_pos[:max_len-1]
        input_ids_pos = input_ids_pos + [cls_id] + [pad_id] * (max_len - 1 - len(input_ids_pos))
        input_ids.append(input_ids_pos)
        neg_ids = example['neg_ids']
        if len(neg_ids) < train_group_size -1:
            num = math.ceil((train_group_size - 1) / len(neg_ids))
            neg_ids = random.sample(neg_ids*num, train_group_size -1)
        else:
            neg_ids = random.sample(neg_ids, train_group_size -1)
        for neg_ids in neg_ids:
            input_ids_neg = query_id_choice + [sep_id] + neg_ids
            if len(input_ids_neg) >= max_len-1:
                input_ids_neg = input_ids_neg[:max_len-1]
            input_ids_neg = input_ids_neg + [cls_id] + [pad_id] * (max_len - 1 - len(input_ids_neg))
            input_ids.append(input_ids_neg)
    return {'input_ids': torch.tensor(input_ids,dtype=torch.long)}
def cross_collate_fn_prev(examples, cls_id, max_len,sep_id, pad_id):
    input_ids = []
    labels = []
    for example in examples:
        query_id_choice = example['query_ids']
        if len(query_id_choice) >= max_len-22:
            #we ask for at least 20 tokens for pos and neg
            query_id_choice = query_id_choice[:max_len-22]
        query_id_choice = query_id_choice + [cls_id]
         # 随机选择pos_ids中的一个元素
        pos_choice = random.choice(example['pos_ids'])
        # 随机选择neg_ids中的一个元素
        neg_choice = random.choice(example['neg_ids'])
        input_ids_pos = query_id_choice + [sep_id] + pos_choice
        input_ids_neg = query_id_choice + [sep_id] + neg_choice
        if len(input_ids_pos) >= max_len-1:
            input_ids_pos = input_ids_pos[:max_len-1]
        input_ids_pos = input_ids_pos + [cls_id] + [pad_id] * (max_len - 1 - len(input_ids_pos))
        if len(input_ids_neg) >= max_len-1:
            input_ids_neg = input_ids_neg[:max_len-1]
        input_ids_neg = input_ids_neg + [cls_id] + [pad_id] * (max_len - 1 - len(input_ids_neg))
        input_ids.append(input_ids_pos)
        input_ids.append(input_ids_neg)
        labels.append(1.0)
        labels.append(0.0)
    return {'input_ids': torch.tensor(input_ids,dtype=torch.long), 'labels': torch.tensor(labels,dtype=torch.float)}
        
def bi_collate_fn(examples, cls_id, max_len, pad_id):
    query_ids = []
    pos_ids = []
    neg_ids = []
    for example in examples:
        query_id_choice = example['query_ids']
        # 在总长度不超过max_len的情况下，在ids尾部添加一个cls_id
        if len(query_id_choice) + 1 <= max_len:
            query_id_choice.append(cls_id)
        else:
            query_id_choice = query_id_choice[:max_len-1] + [cls_id]
        query_ids.append(query_id_choice + [pad_id] * (max_len - len(query_id_choice)))
        # 随机选择pos_ids中的一个元素
        pos_choice = random.choice(example['pos_ids'])
        # 在总长度不超过max_len的情况下，在ids尾部添加一个cls_id
        if len(pos_choice) + 1 <= max_len:
            pos_choice.append(cls_id)
        else:
            pos_choice = pos_choice[:max_len-1] + [cls_id]
        pos_ids.append(pos_choice + [pad_id] * (max_len - len(pos_choice)))
        
        # 随机选择neg_ids中的一个元素
        neg_choice = random.choice(example['neg_ids'])
        # 在总长度不超过max_len的情况下，在ids尾部添加一个cls_id
        if len(neg_choice) + 1 <= max_len:
            neg_choice.append(cls_id)
        else:
            neg_choice = neg_choice[:max_len-1] + [cls_id]
        neg_ids.append(neg_choice + [pad_id] * (max_len - len(neg_choice)))
        
    return {'query_input_ids': torch.tensor(query_ids,dtype=torch.long), 
            'pos_input_ids': torch.tensor(pos_ids,dtype=torch.long), 
            'neg_input_ids': torch.tensor(neg_ids,dtype=torch.long)}
def load_and_tokenize_ds(json_data_file, max_len, cls_id, pad_id,batch_size):
    original_ds = load_dataset('json', data_files=json_data_file)
    tokenized_ds = original_ds.map(tokenization_function, batched=True, remove_columns=original_ds['train'].features,num_proc=4)
    collate_fn = partial(bi_collate_fn, cls_id=cls_id, max_len=max_len, pad_id=pad_id)
    data_loader = DataLoader(tokenized_ds['train'], batch_size=batch_size, collate_fn=collate_fn,shuffle=True, num_workers=4, pin_memory=True)
    return data_loader

def load_and_tokenize_cross_encoder_ds(json_data_file, max_len, cls_id,sep_id, pad_id,batch_size):
    original_ds = load_dataset('json', data_files=json_data_file)
    tokenized_ds = original_ds.map(tokenization_function, batched=True, remove_columns=original_ds['train'].features,num_proc=4)
    collate_fn = partial(cross_collate_fn, cls_id=cls_id, max_len=max_len,sep_id=sep_id, pad_id=pad_id)
    data_loader = DataLoader(tokenized_ds['train'], batch_size=batch_size, collate_fn=collate_fn,shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
    return data_loader

if __name__ == '__main__':
    json_data_file = '/home/yueyulin/tmp/mmarco_chinese_len-0-500.jsonl'
    pad_id = 151334
    cls_id = 151329
    max_len = 512
    sep_id = 151330
    batch_size = 4
    print('pad_id:', pad_id,' cls_id:', cls_id)
    data_loader = load_and_tokenize_cross_encoder_ds(json_data_file, max_len, cls_id,sep_id, pad_id,batch_size)
    for d in data_loader:
        print(d)
        print(d['input_ids'].shape)
        print(d['input_ids'][0].tolist())
        break
    print(len(data_loader))

    