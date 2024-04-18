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
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn

class RwkvBiEncoder():
    def __init__(self,bi_encoder):
        self.bi_encoder = bi_encoder

    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        from tqdm import tqdm
        progress_bar = tqdm(sentences, desc="Encoding")
        embeddings = []
        for sentence in progress_bar:
            embeddings.append(self.bi_encoder.encode_texts(sentence,chunk_size=2048).unsqueeze(0))
        return torch.cat(embeddings, dim=0)

if __name__ == '__main__':
    from src.model_run import RwkvForSequenceEmbedding,create_empty_args,load_embedding_ckpt_and_parse_args,BiEncoder
    from src.model_run import RWKV
    torch.backends.cudnn.benchmark = True
    ckpt = '/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    device = 'cuda'
    tokenizer_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    dtype = torch.bfloat16
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(ckpt, args)
    model = RWKV(args)
    info = model.load_state_dict(w)
    model = model.to(dtype)
    model = model.to(device)
    lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/bi-encoder/add_mlp_in_batch_neg/epoch_0_step_200000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    bi_encoder = BiEncoder(model,lora_path,tokenizer,dtype=dtype,lora_type='lora',add_mlp=True,mlp_dim=1024,lora_r=8,lora_alpha=32,target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],adapter_name='bi_embedding_lora',original_adapter_name='embedding_lora')
    my_eval = RwkvBiEncoder(bi_encoder)
    embeddings = my_eval.encode(['hello world','你好世界'])
    print(embeddings)
    print(embeddings.shape)
    from mteb import MTEB
    from C_MTEB import *
    evaluation = MTEB(tasks=["T2Retrieval"],task_langs=['zh', 'zh-CN'])
    evaluation.run(my_eval)
