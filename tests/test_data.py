import datasets
from datasets import load_from_disk
ds_dir = '/media/yueyulin/data_4t/data/instructIE_zh_ds/train_zh_dataset_ds_512/'
ds = load_from_disk(ds_dir)
import os
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(parent_path)
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
tokenizer = TRIE_TOKENIZER(tokenizer_file)
d = ds[0]
print(d)
input_ids = d['input_ids']
print(input_ids)
print(tokenizer.decode(input_ids))
print('\n')
labels = d['labels']
print(labels)

train_data = '/media/yueyulin/data_4t/data/instructIE_zh_ds'
from data.custom_datasets import read_dataset as read_variable_length_dataset,pad_only_according_data
import torch
from torch.utils.data import DataLoader
train_lengths = [ 256 ,512 ,1024 ,2048]
train_batch_sizes = [ 8 ,4 ,2 ,1]
skip_steps = 0
from data.custom_datasets import MyBatchSampler,pad_and_truncated_according_data
ds = read_variable_length_dataset(train_data,train_lengths)
length_of_dataset = len(ds)
sum_of_batches = sum([(ds.cummulative_sizes[i]-(ds.cummulative_sizes[i-1] if i > 0 else 0))//train_lengths[i] for i in range(len(ds.cummulative_sizes))])
print(sum_of_batches)
batch_size = length_of_dataset // sum_of_batches
print(batch_size)
sampler = MyBatchSampler([i for i in range(len(ds))],batch_size,True,ds.cummulative_sizes,train_batch_sizes,skipped_batches=skip_steps)
train_dataloader = DataLoader(ds,batch_sampler=sampler,collate_fn=pad_only_according_data)
iter_dataloader = iter(train_dataloader)
def first_non_minus_100(l,start_offset):
    for i in range(len(l)-start_offset):
        if l[i+start_offset] != -100:
            return i+start_offset
    return -1

for i in range(4):
    print(f'batch {i}')
    input_ids,labels = next(iter_dataloader)
    input_ids = input_ids.tolist()
    labels = labels.tolist()
    for j in range(len(input_ids)):
        inputs = input_ids[j]
        l = labels[j]
        print(len(inputs))
        print(len(l))
        print(inputs)
        print(l)
        index_of_l = first_non_minus_100(l,0)
        last_of_l = l.index(1)
        
        print(inputs[0:index_of_l+1])
        print(l[index_of_l:last_of_l+1])
        print(tokenizer.decode(inputs[0:index_of_l+1]))
        print(tokenizer.decode(l[index_of_l:last_of_l]))
        print('\n')
    print('------------------------------')