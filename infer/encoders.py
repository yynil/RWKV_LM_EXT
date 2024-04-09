import os
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(f'add {parent_dir} to sys path')
from src.model_ext import RwkvForSequenceEmbedding,load_ckpt_and_parse_args,create_empty_args
from src.model import RWKV
import torch
from torch.cuda import amp
class BiEncoder:
    def __init__(self,
                    base_rwkv,
                    lora_path,
                    tokenizer,
                    lora_type='lora',
                    add_mlp=True,
                    mlp_dim=1024,
                    lora_r=8,
                    lora_alpha=32,
                    target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                    adapter_name='embedding_lora'):
        self.base_rwkv = base_rwkv
        self.add_mlp = add_mlp
        self.mlp_dim = mlp_dim
        args = create_empty_args()
        w = load_ckpt_and_parse_args(base_rwkv,args)
        rwkv = RWKV(args)
        info = rwkv.load_state_dict(w)
        print(f'load model from {base_rwkv},result is {info}')

        self.lora_path = lora_path
        lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        from peft import inject_adapter_in_model
        rwkv = inject_adapter_in_model(lora_config,rwkv,adapter_name=adapter_name)
        print(f'inject lora from {lora_path} to model,result is {rwkv}')

        self.rwkv_embedding = RwkvForSequenceEmbedding(rwkv,add_mlp=add_mlp,output_dim=mlp_dim)
        print(self.rwkv_embedding)
        with torch.no_grad():
            w = torch.load(lora_path)
            info = self.rwkv_embedding.load_state_dict(w,strict=False)
            print(f'load model from {lora_path},result is {info}')
        self.rwkv_embedding = self.rwkv_embedding.bfloat16()
        self.rwkv_embedding = self.rwkv_embedding.cuda()
        self.rwkv_embedding.eval()
        self.tokenizer = tokenizer

    def encode_texts(self,texts):
        MAX_LEN = 4096
        max_len = 0
        input_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)[0:MAX_LEN-1]
            ids.append(self.rwkv_embedding.embedding_id)
            max_len = max(max_len,len(ids))
            input_ids.append(ids)
        input_ids = [ids + [self.rwkv_embedding.pad_id]*(max_len-len(ids)) for ids in input_ids]
        input_ids = torch.tensor(input_ids,dtype=torch.long,device='cuda')
        with torch.no_grad():
            with amp.autocast_mode.autocast(enabled=True,dtype=torch.bfloat16):
                outputs = self.rwkv_embedding.forward(input_ids)
        return outputs
if __name__ == '__main__':
    base_rwkv_model = '/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/bi-encoder/add_mlp_in_batch_neg/epoch_0_step_200000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    tokenizer_file = os.path.join(parent_dir,'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    bi_encoder = BiEncoder(base_rwkv_model,lora_path,tokenizer)
    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = bi_encoder.encode_texts(texts)
    print(outputs)
    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')
    