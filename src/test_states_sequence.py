import os
import sys
parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_parent_dir)
print('add ',parent_parent_dir,' to sys path')
#HF_ENDPOINT=https://hf-mirror.com
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
os.environ['RWKV_TRAIN_TYPE'] = 'infctx'
os.environ["WKV"] = 'fla'
os.environ["WKV"] = ''
model_ckpt_file = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
from src.infctx_module import BlockStateList
from src.model_ext import RwkvStatesForSequenceEmbedding
print('init')
from argparse import Namespace
args = Namespace(n_layer=12, 
                 vocab_size=65536,
                 n_embd=1024, 
                 n_ctx=4096,
                 dropout=0,
                 my_pos_emb=0,
                 pre_ffn=0,
                 head_size_a=64,
                 head_size_divisor=8,
                 chunk_ctx=256,
                 grad_cp = 0,
                 lr_init=4e-4,
                 lr_final=1e-5,
                 head_qk=0,
                 accelerator="GPU")
from src.model_ext import load_ckpt_and_parse_args
w = load_ckpt_and_parse_args(model_ckpt_file,args)
print(args)
model = RwkvStatesForSequenceEmbedding(args)
print(model)
info = model.load_state_dict(w,strict=False)
print(info)
import torch

B,T,C = 2,10,1024

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer_file = os.path.join(parent_parent_dir,'tokenizer','rwkv_vocab_v20230424.txt')
tokenizer = TRIE_TOKENIZER(tokenizer_file)
print(tokenizer)

device = 'cuda'
dtype = torch.bfloat16
model = model.to(device=device,dtype=dtype)
idx = [tokenizer.encode('右侧甲状腺叶切除术'),tokenizer.encode('左睾丸固定')]
print(idx)
idx[0].append(1)
idx[1].append(1)
max_len = 2000
idx[0][len(idx[0]):max_len] = [0]*(max_len-len(idx[0]))
idx[1][len(idx[1]):max_len] = [0]*(max_len-len(idx[1]))
idx = torch.tensor(idx,dtype=torch.long,device=device)
print(idx)
embeddings,last_shift_states,last_wkv_states = model(idx)  
print(embeddings.shape)
print(last_shift_states.shape)
print(last_wkv_states.shape)
print(embeddings)
print(last_shift_states)
print(last_wkv_states)
# from torch import nn
# class StatesCNN(nn.Module):
#     def __init__(self,N,H,h,channel):
#         super(StatesCNN, self).__init__()
#         self.conv1 = nn.Conv3d(N, 32, kernel_size=(3,3,3), stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
#         self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=1)
#         self.fc = nn.Linear(64*H//4*h//4*h//4, channel)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
# N = args.n_layer
# H = args.n_embd//args.head_size_a
# h = args.head_size_a
# states_cnn = StatesCNN(N,H,h,1024)
# states_cnn = states_cnn.to(device=device,dtype=dtype)
# x = states_cnn(last_wkv_states.transpose(0,1))
# print(x.shape)
# exit()
