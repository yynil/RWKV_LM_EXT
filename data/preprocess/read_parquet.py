import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,nargs='+', default='/media/yueyulin/KINGSTON/tmp/parquet/data_nq.jsonl.parquet',help='input data directory')
args = parser.parse_args()

from datasets import Dataset,load_from_disk,concatenate_datasets
import numpy as np
def read_dataset(data_path_list):
    ds = []
    for data_path in data_path_list:
        ds.append(load_from_disk(data_path))
    return concatenate_datasets(ds)

def read_data_len(ds):
    pos_len = []
    neg_len = []
    query_len = []
    from tqdm import tqdm
    progress = tqdm(total=len(ds), desc='read data length')
    for i in range(progress.total):
        query_len.append(len(ds[i]['query']))
        pos_len.extend([len(p) for p in ds[i]['pos']])
        neg_len.extend([len(n) for n in ds[i]['neg']])
        progress.update(1)
    return np.array(query_len),np.array(pos_len),np.array(neg_len)
if __name__ == '__main__':
    ds = read_dataset(args.data_path)
    print(ds)
    query_len,pos_len,neg_len = read_data_len(ds)
    #print the statistics of the lengths of queries, positive and negative samples
    print(f"query_len: {query_len.mean()}, {query_len.std()}, {query_len.max()}, {query_len.min()}")
    print(f"pos_len: {pos_len.mean()}, {pos_len.std()}, {pos_len.max()}, {pos_len.min()}")
    print(f"neg_len: {neg_len.mean()}, {neg_len.std()}, {neg_len.max()}, {neg_len.min()}")

    #print the histogram of the lengths of queries, positive and negative samples
    import matplotlib.pyplot as plt
    plt.hist(query_len,bins=100)
    plt.title('query length')
    plt.show()

    plt.hist(pos_len,bins=100)
    plt.title('pos length')
    plt.show()

    plt.hist(neg_len,bins=100)
    plt.title('neg length')
    plt.show()

