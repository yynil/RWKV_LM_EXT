ckpt = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
states_file = '/tmp/states.pth'
tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'

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
from src.model_run import RWKV,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate
device = 'cuda'
dtype = torch.bfloat16
args = create_empty_args()
w = load_embedding_ckpt_and_parse_args(ckpt, args)
print(args)
model = RWKV(args)
info = model.load_state_dict(w)
model.eval()
print(model)
print(info)

states = torch.load(states_file)
print(states.keys())
states_value = []
n_head = args.n_head
head_size = args.head_size_a
for i in range(args.n_layer):
    key = f'blocks.{i}.att.time_state'
    print(key)
    value = states[key]
    prev_x = torch.zeros(args.n_embd,device=device,dtype=torch.float32)#n_embd 2048 
    prev_states = torch.tensor(value,device=device,dtype=torch.float32)#n_head,head_size,head_size 32,64,64
    prev_ffn = torch.zeros(args.n_embd,device=device,dtype=torch.float32)#n_embd 2048 
    print(prev_x.shape)
    print(prev_states.shape)
    print(prev_ffn.shape)
    states_value.append(prev_x)
    states_value.append(prev_states)
    states_value.append(prev_ffn)
gen_args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.8, top_k = 100, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [0,1], # ban the generation of some tokens
                        token_stop = [0,2], # stop generation whenever you see any token here
                        chunk_len = 256)
cat_char = '🐱'
bot_char = '🤖'
instruction ='曹操送来的木匣妙计中，指示让谁守合淝？可选项：\nA.张辽\nB.李典\nC.乐进\nD.于禁'
input_text = ''
ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
print(ctx)
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER(tokenizer_file)
print(tokenizer.encode(ctx))
model = model.to(dtype)
model = model.to(device)
with torch.no_grad():
    with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
        output = generate(model, ctx,tokenizer, token_count=512, args=gen_args,callback=None,state=states_value)
    print(output)
