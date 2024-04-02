import orjson
import os
import sys
parent_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(parent_parent_path)
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import multiprocessing
import pyarrow.dataset as ds
import pandas as pd
from tqdm import tqdm
import time
from datasets import Dataset
from datasets import concatenate_datasets
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/media/yueyulin/bigdata/data/neg',help='input data directory')
parser.add_argument('--output_path', type=str, default='/media/yueyulin/bigdata/data/parquet/neg',help='output data directory')
parser.add_argument('--num_processes', type=int, default=1, help='number of processes to use')
args = parser.parse_args()
os.makedirs(args.output_path,exist_ok=True)

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            if file.endswith(".jsonl") or file.endswith(".json"):
                all_files.append(os.path.join(root, file))
    return all_files


def read_jsonl(fp,chunksize=10000,tokenizer=None):
    num_error = 0
    num_correct = 0
    results = dict()
    results['pos']=[]
    results['neg']=[]
    results['query']=[]
    for line in fp:
        line = line.strip()
        try:
            data = orjson.loads(line)
            if 'pos' in data and 'neg' in data and 'query' in data:
                query = data['query']
                pos = data['pos']
                neg = data['neg']
                if tokenizer:
                    query = tokenizer.encode(query)
                    pos = [tokenizer.encode(p) for p in pos]
                    neg = [tokenizer.encode(n) for n in neg]
                results['query'].append(query)
                results['pos'].append(pos)
                results['neg'].append(neg)
            num_correct += 1
            if num_correct % chunksize == 0:
                yield results
                results = dict()
                results['pos']=[]
                results['neg']=[]
                results['query']=[]
        except:
            num_error += 1
    if len(results['query']) > 0:
        yield results
    print(f"num_error: {num_error}, num_correct: {num_correct}")

def write_to_dataset(file_path, output_path):
    print(f"\033[91m{os.getpid()} is processing {file_path} and writing to {output_path}\033[00m")
    chunksize = 100000  # adjust this value depending on your available memory
    dataset = None
    tokenizer_file = os.path.join(parent_parent_path,'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    with open(file_path, 'r',encoding='UTF-8') as f:
        for i, chunk in enumerate(read_jsonl(f,chunksize,tokenizer=tokenizer)):
            chunk_dataset = Dataset.from_dict(chunk)
            print(f"\033[93mAdd chunk {i} to {output_path}\033[00m")
            if dataset is None:
                dataset = chunk_dataset
            else:
                dataset = concatenate_datasets([dataset, chunk_dataset])
    # save the final Dataset to disk
    dataset.save_to_disk(output_path)
    print(f"\033[92mFinished processing {file_path} and writing to {output_path}, {dataset}\033[00m")

def main():    
    all_files = get_all_files(args.data_path)
    with multiprocessing.Pool(args.num_processes) as pool:
        data_paths = [os.path.join(args.output_path, f"data_{os.path.basename(filename)}.parquet") for filename in all_files]
        pool.starmap(write_to_dataset, zip(all_files, data_paths))
    # # After all processes complete, read all Parquet files into a single dataset
    # dataset = ds.dataset(data_paths, format="parquet")
    # #save to output_path/final
    # final_dir = os.path.join(args.output_path, "final")
    # os.makedirs(final_dir, exist_ok=True)
    # pq.write_to_dataset(dataset, root_path=final_dir, partition_cols=["filename"])
        
def main_test():
    test_file = '/media/yueyulin/bigdata/data/neg/data_v1_split1/train_15neg/split_1/marco_chinese.jsonl'
    dataset = None
    with open(test_file, 'r',encoding='UTF-8') as f:
        for i, chunk in enumerate(read_jsonl(f,100)):
            chunk_dataset = Dataset.from_dict(chunk)
            if dataset is None:
                dataset = chunk_dataset
            else:
                dataset = concatenate_datasets([dataset, chunk_dataset])
            print(dataset[-1])
            print(dataset)
            if i > 0:
                break
if __name__ == "__main__":
    main()