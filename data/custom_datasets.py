import datasets
from datasets import load_from_disk,concatenate_datasets
from torch.utils.data import DataLoader,Dataset,ConcatDataset,Sampler,BatchSampler
from typing import List, Union, Iterable, Iterator
import os
class FixedLenDataset(Dataset):
    def __init__(self, dataset, fixed_len):
        self.dataset = dataset
        self.fixed_len = fixed_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]
        data['fixed_len'] = self.fixed_len
        return data
    
class MyBatchSampler:

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool,cummulative_sizes,variable_batch_sizes,skip_batches=0) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._sampler = sampler
        self.cummulative_sizes = cummulative_sizes
        self.variable_batch_sizes = variable_batch_sizes
         # Save the arguments
        self.__pl_saved_args = (sampler,batch_size,drop_last,cummulative_sizes,variable_batch_sizes)
        self.world_size = 1
        self.rank = 0
        self.all_batches = sum([(self.cummulative_sizes[i]-(self.cummulative_sizes[i-1] if i > 0 else 0))//(self.variable_batch_sizes[i]*self.world_size) for i in range(len(self.cummulative_sizes))])-self.skip_batches
        self.skip_batches = skip_batches
        #log the arguments 
        print(f"Arguments: {self.__pl_saved_args}")
    @property
    def sampler(self):
        return self._sampler

    def __iter__(self) -> Iterator[List[int]]:
        current_dataset_idx = 0
        rest_batches_in_chunks = [(self.cummulative_sizes[i]-(self.cummulative_sizes[i-1] if i > 0 else 0))//(self.variable_batch_sizes[i]*self.world_size) for i in range(len(self.cummulative_sizes))]
        # Skip the first 'skip_batches' batches
        for _ in range(self.skip_batches):
            while rest_batches_in_chunks[current_dataset_idx] == 0:
                current_dataset_idx += 1
                if current_dataset_idx >= len(self.cummulative_sizes):
                    current_dataset_idx = 0
            rest_batches_in_chunks[current_dataset_idx] -= 1
            current_dataset_idx += 1
            if current_dataset_idx >= len(self.cummulative_sizes):
                current_dataset_idx = 0
        while sum(rest_batches_in_chunks) > 0:
            #get next available data chunk
            while rest_batches_in_chunks[current_dataset_idx] == 0:
                current_dataset_idx += 1
                if current_dataset_idx >= len(self.cummulative_sizes):
                    current_dataset_idx = 0
            #get the next batch
            batch = []
            batch_size = self.variable_batch_sizes[current_dataset_idx]
            for i in range(batch_size):
                batch.append(self.cummulative_sizes[current_dataset_idx] - rest_batches_in_chunks[current_dataset_idx]*batch_size*self.world_size + i+self.rank*batch_size)
            rest_batches_in_chunks[current_dataset_idx] -= 1
            current_dataset_idx += 1
            if current_dataset_idx >= len(self.cummulative_sizes):
                current_dataset_idx = 0
            yield batch

    def __len__(self) -> int:
        return self.all_batches
    
    
    def set_world_size(self, value):
        self.world_size = value
        self.all_batches = sum([(self.cummulative_sizes[i]-(self.cummulative_sizes[i-1] if i > 0 else 0))//(self.variable_batch_sizes[i]*self.world_size) for i in range(len(self.cummulative_sizes))])

    @property
    def sampler(self):
        return self

    def set_epoch(self, epoch):
        pass

import random
import torch

def pad_and_truncated_according_data(features, pad_token_id=0,eos_token_id=1):
    max_len = features[0]['fixed_len']
    query_ids = [feature['query'] for feature in features]
    rand_pos_ids = [min(p, key=lambda x: abs(len(x) - max_len)) for p in [feature['pos'] for feature in features]]
    rand_neg_ids = [min(n, key=lambda x: abs(len(x) - max_len)) for n in [feature['neg'] for feature in features]]
    query_ids = [q[:max_len-1]+[eos_token_id] for q in query_ids]
    rand_pos_ids = [p[:max_len-1]+[eos_token_id] for p in rand_pos_ids]
    rand_neg_ids = [n[:max_len-1]+[eos_token_id] for n in rand_neg_ids]
    query_ids = [q+[pad_token_id]*(max_len-len(q)) for q in query_ids]
    rand_pos_ids = [p+[pad_token_id]*(max_len-len(p)) for p in rand_pos_ids]
    rand_neg_ids = [n+[pad_token_id]*(max_len-len(n)) for n in rand_neg_ids]
    return {'query':torch.tensor(query_ids,dtype=torch.long),
            'positive':torch.tensor(rand_pos_ids,dtype=torch.long),
            'negative':torch.tensor(rand_neg_ids,dtype=torch.long)}    

def read_dataset(parent_dir,max_lengths):
    fixed_length_datasets = []
    for max_len in max_lengths:
        directories = []
        for root, dirs, files in os.walk(parent_dir,topdown=False):
            #check if the dir name ends with max_len
            for name in dirs:
                if name.endswith(str(max_len)):
                    directories.append(os.path.join(root, name))
        print(directories)

        conated_dataset = concatenate_datasets([load_from_disk(d) for d in directories])
        print(conated_dataset)
        fixed_dataset = FixedLenDataset(conated_dataset, max_len)
        print(len(fixed_dataset))
        fixed_length_datasets.append(fixed_dataset)
    dataset = ConcatDataset(fixed_length_datasets)
    return dataset
def cross_encoder_pad_and_truncated_according_data(features,pad_token_id=0,eos_token_id=1,sep_token_id=2):
    max_len = features[0]['fixed_len']
    query_ids = [feature['query'] for feature in features]
    rand_pos_ids = [min(p, key=lambda x: abs(len(x) - max_len)) for p in [feature['pos'] for feature in features]]
    rand_neg_ids = [min(n, key=lambda x: abs(len(x) - max_len)) for n in [feature['neg'] for feature in features]]
    #create the data 
    #{
    #    "input_ids": to [query_ids sep_token_id rand_pos_ids eos_token_id pad_token_id*(max_len-pos_len)]+[query_ids sep_token_id rand_neg_ids eos_token_id pad_token_id*(max_len-neg_len)] 
    #    "labels": [1,0]
    #}
    input_ids = []
    labels = []
    max_len = 0
    for i in range(len(query_ids)):
        pos_ids = rand_pos_ids[i]
        neg_ids = rand_neg_ids[i]
        q_ids = query_ids[i]
        q_input_ids = q_ids + [sep_token_id] + pos_ids + [eos_token_id]
        if len(q_input_ids) > max_len:
            max_len = len(q_input_ids)
        input_ids.append(q_input_ids)
        labels.append(1)
        q_input_ids = q_ids + [sep_token_id] + neg_ids + [eos_token_id]
        if len(q_input_ids) > max_len:
            max_len = len(q_input_ids)
        input_ids.append(q_input_ids)
        labels.append(0)
    #pad the input_ids to max_len even the max_len is 2 bytes
    input_ids = [q+[pad_token_id]*(max_len-len(q)) for q in input_ids]
    return {'input_ids':torch.tensor(input_ids,dtype=torch.long),
            'labels':torch.tensor(labels,dtype=torch.long)}
if __name__ == '__main__':
    import os
    parent_dir = '/media/yueyulin/KINGSTON/tmp/parquet_chuncked/'
    max_lengths = [128,256]
    dataset = read_dataset(parent_dir,max_lengths)
    print(len(dataset))
    print(dataset.cummulative_sizes)
    from torch.utils.data import SequentialSampler
    length_of_dataset = len(dataset)
    variable_batch_sizes = [4,2]
    sum_of_batches = sum([(dataset.cummulative_sizes[i]-(dataset.cummulative_sizes[i-1] if i > 0 else 0))//variable_batch_sizes[i] for i in range(len(dataset.cummulative_sizes))])
    print(sum_of_batches)
    batch_size = length_of_dataset // sum_of_batches
    print(batch_size)
    sampler = MyBatchSampler([i for i in range(length_of_dataset)],batch_size,True,dataset.cummulative_sizes,variable_batch_sizes)
    dataloader = DataLoader(dataset, batch_sampler=sampler,collate_fn=cross_encoder_pad_and_truncated_according_data)
    print(len(dataloader))
    idx = 0
    sampler.set_world_size(8)
    print(len(dataloader))
    for batch in dataloader:
        if idx % 100 == 0:
            print("minibatch:",idx)
            print(batch['input_ids'].shape)
            print(batch['labels'].shape)
        idx += 1


    