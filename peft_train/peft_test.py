import argparse
import torch

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

def generate_text(model,instruction,input,tokenizer):
    gen_args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.96, top_k = 20, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 512)
    cat_char = '🐱'
    bot_char = '🤖'
    ctx = f'{cat_char}:{instruction}\n{input}\n{bot_char}:'
    with torch.no_grad():
        with torch.autocast(enabled=True,device_type='cuda',dtype=torch.bfloat16):
            output = generate(model, ctx,tokenizer, token_count=128, args=gen_args,callback=None)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test the lora/pissa/state finetune generation/biencoder")
    parser.add_argument("--base_model", type=str, help="Base model file",default='/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth')
    parser.add_argument("--chat_ckpt", type=str, help="chat_ckpt",default='/media/yueyulin/data_4t/models/lora_rwkv/epoch_7/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth')
    parser.add_argument("--tokenizer_file", type=str, help="tokenizer_file",default='/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt')
    parser.add_argument("--biencoder_ckpt", type=str, help="bi_encoder ckpt",default='/media/yueyulin/data_4t/models/lora/biencoder/epoch_1_step_430000/RWKV-x060-World-1B6-v2_rwkv_lora.pth')
    parser.add_argument("--chat_lora_alpha", type=float, default=32, help="lora_alpha")
    parser.add_argument("--chat_lora_r", type=int, default=8, help="lora_r")
    parser.add_argument("--chat_targets",nargs='+',default=["att","ffn"], help="chat_targets")
    parser.add_argument("--biencoder_lora_alpha", type=float, default=32, help="lora_alpha")
    parser.add_argument("--biencoder_lora_r", type=int, default=8, help="lora_r")
    parser.add_argument("--biencoder_targets",nargs='+',default=['emb','ffn.key','ffn.value','ffn.receptance'], help="biencoder_targets")
    args = parser.parse_args()
    print(args)

    #setup env and load base model and tokenizer    
    setup_env()
    from src.model_run import RWKV,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate
    device = 'cuda'
    dtype = torch.bfloat16
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(args.tokenizer_file)
    model = load_base_model(args.base_model)
    model = model.to(device=device,dtype=dtype)
    from src.layers import inject_lora_adapter_with_state_dict,set_adapter
    # load the chat lora
    if args.chat_ckpt:
        chat_lora_state_dict = torch.load(args.chat_ckpt, map_location='cpu')
        chat_lora_name = 'chat_lora_adapter'
        inject_lora_adapter_with_state_dict(
            model,
            chat_lora_name,
            chat_lora_state_dict,
            args.chat_lora_r,
            args.chat_lora_alpha,
            args.chat_targets)
        print(model)
    if args.biencoder_ckpt:
        biencoder_state_dict = torch.load(args.biencoder_ckpt, map_location='cpu')
        biencoder_lora_name = 'biencoder_lora_adapter'
        inject_lora_adapter_with_state_dict(
            model,
            biencoder_lora_name,
            biencoder_state_dict,
            args.biencoder_lora_r,
            args.biencoder_lora_alpha,
            args.biencoder_targets)
        print(model)    
        from src.model_run import RwkvForSequenceEmbedding
        add_mlp = 'dense.weight' in biencoder_state_dict
        output_dim = -1
        should_delete_head = False
        if add_mlp:
            output_dim = biencoder_state_dict['dense.weight'].shape[0]
        print(f'RWKV Embedding model add_mlp = {add_mlp} output_dim = {output_dim}')
        rwkv_embedding = RwkvForSequenceEmbedding(model,
                                                  add_mlp=add_mlp,
                                                  output_dim=output_dim,
                                                  should_delete_head=should_delete_head)
        if add_mlp:
            rwkv_embedding.dense.weight.data = biencoder_state_dict['dense.weight'].to(device=device,dtype=dtype)
            rwkv_embedding.dense.bias.data = biencoder_state_dict['dense.bias'].to(device=device,dtype=dtype)
        print(rwkv_embedding)
        def encode_texts(text,chunk_size=1024):
            input_ids =  tokenizer.encode(text)
            input_ids.append(rwkv_embedding.embedding_id)
            state = None
            offset = 0
            while offset < len(input_ids):
                chunk = input_ids[offset:offset+chunk_size]
                with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
                    outputs,state = rwkv_embedding(torch.tensor(chunk,dtype=torch.long,device=device),state=state)
                offset += len(chunk)

            return outputs
    if args.chat_ckpt:
        set_adapter(model= model,adapter_name=chat_lora_name)
    
        instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
        input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问。'
        output = generate_text(model,instruction,input_text,tokenizer)
        print(output)

    if args.biencoder_ckpt:
        set_adapter(model= model,adapter_name=biencoder_lora_name)
        texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
        outputs = [encode_texts(text) for text in texts]
        print(outputs)
        from sentence_transformers.util import pairwise_cos_sim
        for qid in range(len(texts)):
            query = outputs[qid]
            for i in range(len(texts)):
                if i != qid:
                    print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

            print('-----------------------')

    if args.chat_ckpt:
        set_adapter(model= model,adapter_name=chat_lora_name)
    
        instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
        input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问。'
        output = generate_text(model,instruction,input_text,tokenizer)
        print(output)

    if args.biencoder_ckpt:
        set_adapter(model= model,adapter_name=biencoder_lora_name)
        texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
        outputs = [encode_texts(text) for text in texts]
        print(outputs)
        from sentence_transformers.util import pairwise_cos_sim
        for qid in range(len(texts)):
            query = outputs[qid]
            for i in range(len(texts)):
                if i != qid:
                    print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

            print('-----------------------')


    if args.chat_ckpt:
        set_adapter(model= model,adapter_name=chat_lora_name)
    
        instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
        input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问。'
        output = generate_text(model,instruction,input_text,tokenizer)
        print(output)

    if args.biencoder_ckpt:
        set_adapter(model= model,adapter_name=biencoder_lora_name)
        texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
        outputs = [encode_texts(text) for text in texts]
        print(outputs)
        from sentence_transformers.util import pairwise_cos_sim
        for qid in range(len(texts)):
            query = outputs[qid]
            for i in range(len(texts)):
                if i != qid:
                    print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

            print('-----------------------')