import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/media/yueyulin/bigdata/data/parquet/neg',help='input data directory')
args = parser.parse_args()

import pyarrow.parquet as pq
import os
if __name__ == '__main__':
    from datasets import Dataset,load_from_disk

    ds = load_from_disk(args.data_path)
    print(ds)