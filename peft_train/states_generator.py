ckpt = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096-YIDUPISSA.pth'

states_file = '/home/yueyulin/models/trained/yidu_pissa_states_7B/20240706-000324/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096-YIDUPISSA.pth.pth'
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
    prev_states = torch.tensor(value,device=device,dtype=torch.float).transpose(1,2)#n_head,head_size,head_size 32,64,64
    prev_ffn = torch.zeros(args.n_embd,device=device,dtype=torch.float)#n_embd 2048 
    print(prev_x.shape)
    print(prev_states.shape)
    print(prev_ffn.shape)
    states_value.append(prev_x)
    states_value.append(prev_states)
    states_value.append(prev_ffn)
gen_args = PIPELINE_ARGS(temperature = 1.0, top_p = 0, top_k = 0, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 512)
cat_char = '🐱'
bot_char = '🤖'
instruction ='请把我给你的原始词，转换成标准医疗操作代码和标准词。比如我说“横结肠造口还纳术”，你要回答“代码：46.5204 标准词：横结肠造口闭合术”。寻找和标准医疗操作代码最接近的标准词，不能凭空编造。如果有多个标准词，用##分隔。如输入”右眼白内障超声乳化抽吸术+人工晶体置入术“，则输出”代码：13.7000 标准词：置入人工晶状体##代码：13.4100 标准词：白内障晶状体乳化和抽吸“'
input_text = '左甲状腺部分切除+右甲状腺腺叶切除术'
ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
print(ctx)
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER(tokenizer_file)

model = model.to(dtype)
model = model.to(device)
with torch.no_grad():
    with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
        output = generate(model, ctx,tokenizer, token_count=128, args=gen_args,callback=None,state=states_value)
    print(output)
