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

def load_base_model(base_model):
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(base_model, args)
    print(args)
    model = RWKV(args)
    info = model.load_state_dict(w)
    return model
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test the lora/pissa/state finetune generation/biencoder")
    parser.add_argument("--base_model", type=str, help="Base model file",default='/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth')
    parser.add_argument("--chat_ckpt", type=str, help="chat_ckpt",default='/media/yueyulin/data_4t/models/pissa_r64/epoch_0/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth')
    parser.add_argument("--pissa_dict", type=str, help="pissa_dict",default='/media/yueyulin/data_4t/models/pissa_r64/init_pissa.pth')
    parser.add_argument("--tokenizer_file", type=str, help="tokenizer_file",default='/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt')
    parser.add_argument("--chat_lora_alpha", type=float, default=64, help="lora_alpha")
    parser.add_argument("--chat_lora_r", type=int, default=64, help="lora_r")
    parser.add_argument("--chat_targets",nargs='+',default=["att","ffn"], help="chat_targets")
    args = parser.parse_args()
    print(args)
    setup_env()
    from src.model_run import RWKV,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate_beamsearch
    device = 'cuda'
    dtype = torch.bfloat16
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(args.tokenizer_file)
    model = load_base_model(args.base_model)
    model = model.to(device=device,dtype=dtype)
    from src.layers import inject_lora_adapter_with_state_dict,set_adapter
    
    if args.chat_ckpt:
        chat_lora_state_dict = torch.load(args.chat_ckpt, map_location='cpu')
        pissa =  torch.load(args.pissa_dict, map_location='cpu')
        chat_lora_name = 'chat_lora_adapter'
        inject_lora_adapter_with_state_dict(
            model,
            chat_lora_name,
            chat_lora_state_dict,
            args.chat_lora_r,
            args.chat_lora_alpha,
            args.chat_targets,
            pissa_dict=pissa)
        print(model)
        set_adapter(model,chat_lora_name)
        instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
        input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问。'
        cat_char = '🐱'
        bot_char = '🤖'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type='cuda',dtype=torch.bfloat16):
                results = generate_beamsearch(
                    model, 
                    ctx,tokenizer, 
                    token_count=10,
                    num_beams=5,
                    return_num_sequences=5,
                    num_group=5,
                    do_sample=True)
                import math
                for score, output,beam_idx in results:
                    print(f'{math.exp(score.item())}: {tokenizer.decode(output.tolist())} beam_idx={beam_idx}')
