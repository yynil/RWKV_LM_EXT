import os
os.environ['HF_DATASETS_CACHE'] = '/media/yueyulin/plugable/cache'
#os.environ['HF_ENDPOINT']='https://hf-mirror.com'

from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.zh")
print(ds)