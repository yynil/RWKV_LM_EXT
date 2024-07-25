import torch
import argparse
def setup_env():
    import os
    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    import sys
    sys.path.append(parent_path)
    print(f'add path: {parent_path} to sys.path')
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    os.environ['WKV'] = ''
    os.environ['RWKV_TRAIN_TYPE'] = ''
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    os.environ['NO_CUDA'] = '0'
setup_env()
from transformers import AutoTokenizer
from src.model_encoder_run import RwkvEncoder
from src.model_run import create_empty_args,load_embedding_ckpt_and_parse_args
def load_base_model(base_model):
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(base_model, args)
    print(args)
    args.emb_id = 151329
    args.pad_id = 151334
    args.mask_id = 151330
    model = RwkvEncoder(args)
    info = model.load_state_dict(w)
    print(info)
    return model

parser = argparse.ArgumentParser("Test MLM model")
parser.add_argument("--model_file",type=str,default='/media/yueyulin/KINGSTON/models/macro_zh_bi_encoder/trainable_model/epoch_21/RWKV-x060-MLM-ctx4096.pth.pth')
args = parser.parse_args() 
print(args)
model = load_base_model(args.model_file)

args.emb_id = 151329
args.pad_id = 151334
args.mask_id = 151330
device = 'cuda'
model = model.to(device=device,dtype=torch.bfloat16)    

from mteb import MTEB
from C_MTEB import *
from tqdm import tqdm
def encode_texts(args, model, device, texts, tokenizer,MAX_LEN=512):
    texts_idx = [tokenizer.encode(text,add_special_tokens=False) for text in texts]
    for text_idx in texts_idx:
        text_idx = text_idx[:MAX_LEN]
        text_idx.append(args.emb_id)
    max_len = max([len(text_idx) for text_idx in texts_idx])
    texts_idx = [text_idx + [args.pad_id]*(max_len-len(text_idx)) for text_idx in texts_idx]
    import time 
   
    input_ids = torch.tensor(texts_idx,dtype=torch.long,device=device)
    from sentence_transformers.util import cos_sim as sim_fn
    with torch.no_grad():
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            start_time = time.time()
            embs = model.encode_sentence(input_ids)
            end_time = time.time()
            return embs
tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)

class MyModel():
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        embeddings = []
        step = 4
        progress = tqdm(total=len(sentences),desc='encode')
        for i in range(0,len(sentences)-step,step):
            results = encode_texts(args, model, device, sentences[i:i+step], tokenizer)
            for emb in results:
                emb = emb.cpu().float()
                embeddings.append(emb)
            progress.update(step)
        progress.close()
        return embeddings

mymodel = MyModel()
evaluation = MTEB(tasks=["MMarcoRetrieval"])
evaluation.run(mymodel,verbosity=1)