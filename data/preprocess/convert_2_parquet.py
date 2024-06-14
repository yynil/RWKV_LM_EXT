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
import colorama
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/media/yueyulin/bigdata/data/neg',help='input data directory')
parser.add_argument('--output_path', type=str, default='/media/yueyulin/bigdata/data/parquet/neg_chunked',help='output data directory')
parser.add_argument('--num_processes', type=int, default=1, help='number of processes to use')
args = parser.parse_args()
os.makedirs(args.output_path,exist_ok=True)
import bisect
maxium_lens = [128,256,512,1024,2048,4096]

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            if file.endswith(".jsonl") or file.endswith(".json"):
                all_files.append(os.path.join(root, file))
    return all_files


def check_array_dimension(arr):
    if isinstance(arr, list):
        if isinstance(arr[0], str):
            return 1
    elif isinstance(arr, str):
        return 0
    return -1

def read_jsonl_sized_chunks(fp,chunksize=10000,tokenizer=None):
    num_error = 0
    num_correct = 0
    results = []
    results = [dict() for _ in range(len(maxium_lens)+1)]
    for result in results:
        result['pos']=[]
        result['neg']=[]
        result['query']=[]

    for line in fp:
        line = line.strip()
        try:
            data = orjson.loads(line)
            if 'pos' in data and 'neg' in data and 'query' in data:
                query = data['query']
                pos = data['pos']
                dimension = check_array_dimension(pos)
                if dimension == -1:
                    num_error += 1
                    continue
                elif dimension == 0:
                    pos = [pos]
                neg = data['neg']
                dimension = check_array_dimension(neg)
                if dimension == -1:
                    num_error += 1
                    continue
                elif dimension == 0:
                    neg = [neg]

                if tokenizer:
                    query = tokenizer.encode(query)
                    pos = [tokenizer.encode(p) for p in pos]
                    neg = [tokenizer.encode(n) for n in neg]

                """
                The following is to copy the data to larger size buckets
                segmented_pos = [[] for _ in range(len(maxium_lens)+1)]
                segemented_neg = [[] for _ in range(len(maxium_lens)+1)]
                for p in pos:
                    length = len(p)
                    inserted_index = bisect.bisect_left(maxium_lens,length)
                    for index in range(inserted_index,len(maxium_lens)+1):
                        segmented_pos[index].append(p)
                for n in neg:
                    length = len(n)
                    inserted_index = bisect.bisect_left(maxium_lens,length)
                    for index in range(inserted_index,len(maxium_lens)+1):
                        segemented_neg[index].append(n)
                query_index = bisect.bisect_left(maxium_lens,len(query))
                for index in range(len(maxium_lens)+1):
                    if index >= query_index and len(segmented_pos[index]) > 0 and len(segemented_neg[index]) > 0:
                        results[index]['query'].append(query)
                        results[index]['pos'].append(segmented_pos[index])
                        results[index]['neg'].append(segemented_neg[index])
                """
                """
                The following is to just keep the data into the largest size bucket
                """
                query_index = bisect.bisect_left(maxium_lens,len(query))
                larget_pos_index = 0
                larget_neg_index = 0
                for p in pos:
                    length = len(p)
                    inserted_index = bisect.bisect_left(maxium_lens,length)
                    if inserted_index > larget_pos_index:
                        larget_pos_index = inserted_index
                #we only have one pos and multiple neg
                for n in neg:
                    length = len(n)
                    inserted_index = bisect.bisect_left(maxium_lens,length)
                    bucket_index = max(inserted_index,larget_pos_index,query_index)
                    results[bucket_index]['query'].append(query)
                    results[bucket_index]['pos'].append(pos)
                    results[bucket_index]['neg'].append([n])
                #     if inserted_index > larget_neg_index:
                #         larget_neg_index = inserted_index
                # bucket_index = max(larget_pos_index,larget_neg_index,query_index)
                # results[bucket_index]['query'].append(query)
                # results[bucket_index]['pos'].append(pos)
                # results[bucket_index]['neg'].append(neg)
            num_correct += 1
            if num_correct % chunksize == 0:
                yield results
                results = [dict() for _ in range(len(maxium_lens)+1)]
                for result in results:
                    result['pos']=[]
                    result['neg']=[]
                    result['query']=[]
        except Exception as e:
            import traceback
            traceback.print_exc()
            num_error += 1
    yield results
    print(f"num_error: {num_error}, num_correct: {num_correct} of file {fp.name}")

def write_to_chunked_dataset(file_path, output_path):
    print(f"\033[91m{os.getpid()} is processing {file_path} and writing to {output_path}\033[00m")
    output_paths = [os.path.join(output_path,f"max_len_{len}") for len in maxium_lens]
    output_paths.append(os.path.join(output_path,"max_len_others"))

    chunksize = 100000  # adjust this value depending on your available memory
    chunked_datasets = [None for _ in range(len(output_paths))]
    tokenizer_file = os.path.join(parent_parent_path,'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    with open(file_path, 'r',encoding='UTF-8') as f:
        chunk_index = 0
        for chunks in read_jsonl_sized_chunks(f,chunksize,tokenizer=tokenizer):
            tmpDatasets = [Dataset.from_dict(chunk) for chunk in chunks]
            #concatenate with previous ones
            print(colorama.Fore.RED+f"Processing {file_path} chunk {chunk_index}"+colorama.Style.RESET_ALL)
            for i in range(len(tmpDatasets)):
                if chunked_datasets[i] is None:
                    chunked_datasets[i] = tmpDatasets[i]
                else:
                    chunked_datasets[i] = concatenate_datasets([chunked_datasets[i],tmpDatasets[i]])
            chunk_index += 1
    #write to the disk
    for i in range(len(output_paths)):
        print(f"\033[92mFinished processing {file_path} and writing to {output_paths[i]}, {chunked_datasets[i]}\033[00m")
        chunked_datasets[i].save_to_disk(output_paths[i])
    

def main():    
    all_files = get_all_files(args.data_path)
    with multiprocessing.Pool(args.num_processes) as pool:
        data_paths = [os.path.join(args.output_path, f"data_{os.path.basename(filename)}.parquet") for filename in all_files]
        pool.starmap(write_to_chunked_dataset, zip(all_files, data_paths))
    # # After all processes complete, read all Parquet files into a single dataset
    # dataset = ds.dataset(data_paths, format="parquet")
    # #save to output_path/final
    # final_dir = os.path.join(args.output_path, "final")
    # os.makedirs(final_dir, exist_ok=True)
    # pq.write_to_dataset(dataset, root_path=final_dir, partition_cols=["filename"])
        

if __name__ == "__main__":
    main()