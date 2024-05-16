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
    prev_x = torch.zeros(args.n_embd,device=device,dtype=torch.float)#n_embd 2048 
    prev_states = torch.tensor(value,device=device,dtype=torch.float)#n_head,head_size,head_size 32,64,64
    prev_ffn = torch.zeros(args.n_embd,device=device,dtype=torch.float)#n_embd 2048 
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
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 256)
cat_char = '🐱'
bot_char = '🤖'
instruction ='根据给定的短文，回答以下问题：动物的器官感觉与人的相比有什么不同?'
input_text = '许多动物的某些器官感觉特别灵敏，它们能比人类提前知道一些灾害事件的发生，例如，海洋中的水母能预报风暴，老鼠能事先躲避矿井崩塌或有害气体，等等。地震往往能使一些动物的某些感觉器官受到刺激而发生异常反应。如一个地区的重力发生变异，某些动物可能通过它们的平衡器官感觉到；一种振动异常，某些动物的听觉器官也许能够察觉出来。地震前地下岩层早已在逐日缓慢活动，而断层面之间又具有强大的摩擦力。这种摩擦力会产生一种低于人的听觉所能感觉到的低频声波。人对每秒20次以上的声波才能感觉到，而动物则不然。那些感觉十分灵敏的动物，在感触到这种低声波时，便会惊恐万状，以至出现冬蛇出洞、鱼跃水面等异常现象。'
ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
print(ctx)
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER(tokenizer_file)
print(tokenizer.encode(ctx))
model = model.to(dtype)
model = model.to(device)
with torch.no_grad():
    with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
        output = generate(model, ctx,tokenizer, token_count=128, args=gen_args,callback=None,state=states_value)
    print(output)
