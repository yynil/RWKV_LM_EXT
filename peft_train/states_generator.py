ckpt = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'

states_file = '/media/yueyulin/data_4t/models/states_tuning/large_lr/trainable_model/epoch_0/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
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
instruction ='你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。'
input_text = '{\"input\":\"6 月 17 日，广发证券研报指出，近期大飞机各项进展持续推进。6 月 14 日，东航 C919 机型开启第四条商业定期航线——上海虹桥往返广州白云。\
\
工业和信息化部、国家自然科学基金委员会 6 月 14 日签署合作协议，共同设立大飞机基础研究联合基金。\
\
全球积压飞机订单超 1.4 万架，当前全球航空业因零部件供应短缺、交付周期变长等问题面临供应链威胁，或为国内航空航发产业链相关企业带来航空出海业务新增量。\",\
\"schema\":[\"地理位置\",\"组织机构\",\"气候类型\",\"时间\"]}'
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
