ckpt = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
lora_file = '/media/yueyulin/data_4t/models/lora/epoch_0/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
target_modules = ['emb','ffn.key','ffn.value','ffn.receptance','att.key','att.value','att.receptance']
lora_r = 8
lora_alpha = 32
lora_dropout = 0
is_lora = True
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
if is_lora:
    from peft import LoraConfig
    lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=lora_dropout)
    from peft import inject_adapter_in_model
    model = inject_adapter_in_model(lora_config,model,adapter_name='sft_lora')
    print(model)
    states = torch.load(lora_file)
    print(states.keys())
    info = model.load_state_dict(states,strict=False)
    print(info)
    model.eval()
states_value = None
gen_args = PIPELINE_ARGS(temperature = 1, top_p = 0.96, top_k = 20, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 512)
cat_char = '🐱'
bot_char = '🤖'
instruction ='根据给定的短文，用最简洁的语言回答以下问题：PCIe成为新的个人电脑主板标准的最主要的原因是什么?'
input_text = """应用与前景
技嘉GV-NX62TC256D8显卡，采用PCI Express x16插槽

在2005年，PCIe已近乎成为新的个人电脑主板标准。关于此有不少评论，但最基本的原因是它对于软件开发者完全透明——为PCI所设计的操作系统可以不做任何代码修改来启动PCIe设备。其二，它能增强系统性能，还有强有力的品牌认知。各类网卡、声卡、显卡，以及当下的NVMe固态硬盘都使用了PCIe标准。下面为主流的使用PCIe 的外设产品。
显卡

大部分新型的AMD或NVIDIA显卡都使用PCIe标准。NVIDIA在它新开发的SLI上采用PCIe的高速数据传输，这使得两块相同芯片组显卡可同时工作于一台电脑之上。AMD公司也基于PCIe开发一种两个GPU一同运作的技术，称为CrossFire。
硬盘
当下主流的固态硬盘接口有M.2、U.2、PCIe、SATA、SATA Express、SAS等。M.2和U.2可选PCIe接口[14]。NVMe协议是目前最高效的PCIe SSD协议标准。  
"""
ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
print(ctx)
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER(tokenizer_file)
print(len(tokenizer.encode(ctx)))
model = model.to(dtype)
model = model.to(device)
with torch.no_grad():
    with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
        output = generate(model, ctx,tokenizer, token_count=256, args=gen_args,callback=None,state=None)
    print(output)
