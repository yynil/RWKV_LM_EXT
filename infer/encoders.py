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
from src.model_run import RWKV,RwkvForClassification,RwkvForSequenceEmbedding,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate,generate_beamsearch
import torch
from torch.cuda import amp
from src.layers import inject_lora_adapter_with_state_dict,set_adapter


class BiCrossFusionEncoder:
    """
    This encoder is to fuse the bi-encoder and cross-encoder with the same rwkv base model injected with 2 sets of lora adapters.
    We have 2 assumptions here:
    1. The bi-encoder and cross-encoder share the same base model
    2. The lora types of bi-encoder and cross-encoder are the same
    This class instance is not thread-safe since we need to switch the adapter name before encoding.
    """
    def __init__(self,
                 base_rwkv,
                 bi_lora_path,
                 cross_lora_path,
                 chat_lora_path,
                 tokenizer,
                 ce_lora_r=8,
                 ce_lora_alpha=32,
                 be_lora_r=8,
                 be_lora_alpha=32,
                 chat_lora_r=8,
                 chat_lora_alpha=8,
                 target_ce_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                 target_be_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                 target_chat_modules=['att','ffn'],
                 cross_adapter_name='cross_encoder_lora',
                 bi_adapter_name='bi_embedding_lora',
                 chat_adapter_name='chat_lora',
                 sep_token_id = 2,
                 chat_pissa_path = None,
                 device = 'cuda',
                 dtype = torch.bfloat16
                 ) -> None:
        self.base_rwkv = base_rwkv
        args = create_empty_args()
        w = load_embedding_ckpt_and_parse_args(base_rwkv,args)
        rwkv = RWKV(args)
        info = rwkv.load_state_dict(w)
        print(f'load model from {base_rwkv},result is {info}')
        should_delete_head = False
        #load cross encoder and inject cross adapter
        cross_encoder_dict = torch.load(cross_lora_path,map_location='cpu')
        inject_lora_adapter_with_state_dict(
            rwkv,
            cross_adapter_name,
            cross_encoder_dict,
            ce_lora_r,
            ce_lora_alpha,
            targets=target_ce_modules,
        )
        self.cross_encoder = RwkvForClassification(rwkv,should_delete_head=should_delete_head)
        self.cross_encoder.score.weight.data = cross_encoder_dict['score.weight']
        print(f'load model from {cross_lora_path},result is {info}')
        del cross_encoder_dict
        #load bi encoder and inject bi adapter
        bi_encoder_dict = torch.load(bi_lora_path,map_location='cpu')
        inject_lora_adapter_with_state_dict(
            rwkv,
            bi_adapter_name,
            bi_encoder_dict,
            be_lora_r,
            be_lora_alpha,
            targets=target_be_modules,
        )
        add_mlp = 'dense.weight' in bi_encoder_dict
        output_dim = 0
        if add_mlp:
            output_dim = bi_encoder_dict['dense.weight'].shape[0]
        print(f'RWKV Embedding model add_mlp = {add_mlp} output_dim = {output_dim}')
        self.bi_encoder = RwkvForSequenceEmbedding(rwkv,add_mlp=add_mlp,output_dim=output_dim,should_delete_head=should_delete_head)
        if add_mlp:
            self.bi_encoder.dense.weight.data = bi_encoder_dict['dense.weight']
            self.bi_encoder.dense.bias.data = bi_encoder_dict['dense.bias']
        #load chat lora and inject chat adapter
        chat_lora_dict = torch.load(chat_lora_path,map_location='cpu')
        pissa =  torch.load(chat_pissa_path, map_location='cpu') if chat_pissa_path else None
        inject_lora_adapter_with_state_dict(
            rwkv,
            chat_adapter_name,
            chat_lora_dict,
            chat_lora_r,
            chat_lora_alpha,
            targets=target_chat_modules,
            pissa_dict=pissa
        )
        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id
        self.rwkv = rwkv
        
        #move to device
        self.cross_encoder = self.cross_encoder.to(device=device,dtype=dtype)
        self.bi_encoder = self.bi_encoder.to(device=device,dtype=dtype)
        self.rwkv = self.rwkv.to(device=device,dtype=dtype)
        self.rwkv.eval()
        self.cross_adapter_name = cross_adapter_name
        self.bi_adapter_name = bi_adapter_name
        self.chat_adapter_name = chat_adapter_name
        self.device = device
        self.dtype = dtype

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        set_adapter(model=self.rwkv,adapter_name=adapter_name)

    def encode_text(self,text,chunk_size=1024):
        input_ids =  tokenizer.encode(text)
        input_ids.append(self.bi_encoder.embedding_id)
        state = None
        offset = 0
        while offset < len(input_ids):
            chunk = input_ids[offset:offset+chunk_size]
            with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
                outputs,state = self.bi_encoder(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
            offset += len(chunk)

        return outputs.tolist()

    def encode_texts(self,texts):
        self.set_adapter(self.bi_adapter_name)
        return [self.encode_text(text) for text in texts]
    
    def cross_encode_text(self,text_a, text_b):
        text_a_ids = self.tokenizer.encode(text_a)
        text_b_ids = self.tokenizer.encode(text_b)
        input_ids = text_a_ids+[self.sep_token_id]+text_b_ids+[self.cross_encoder.class_id]
        offset = 0
        state = None
        with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
            while offset < len(input_ids):
                chunk = input_ids[offset:offset+1024]
                output,state = self.cross_encoder(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
                offset += len(chunk)
        return output.item()

    def cross_encode_texts(self,texts_a, texts_b):
        assert len(texts_a) == len(texts_b)
        self.set_adapter(self.cross_adapter_name)
        outputs = []
        for text_a,text_b in zip(texts_a,texts_b):
            outputs.append(self.cross_encode_text(text_a,text_b))
        return outputs
    
    def beam_generate(self,instruction,input_text,token_count=128,num_beams=5,return_num_sequences=5,num_group=5,do_sample=True,is_sum_logprobs=True,length_penalty=0.6):
        self.set_adapter(self.chat_adapter_name)
        cat_char = '🐱'
        bot_char = '🤖'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
                results = generate_beamsearch(
                    self.rwkv, 
                    ctx,self.tokenizer, 
                    token_count=token_count,
                    num_beams=num_beams,
                    return_num_sequences=return_num_sequences,
                    num_group=num_group,
                    do_sample=do_sample,
                    is_sum_logprobs=is_sum_logprobs,
                    length_penalty=length_penalty)
        import math
        results = [(tokenizer.decode(output.tolist()),math.exp(score.item()),beam_idx) for score, output,beam_idx in results]   
        return results

    def sampling_generate(self,instruction,input_text,token_count=128,
                          temperature=1.0,
                          top_p=0,
                          top_k=0,
                          alpha_frequency=0.25,
                          alpha_presence=0.25,
                          alpha_decay=0.996,
                          token_stop=[0,1]):
        self.set_adapter(self.chat_adapter_name)
        gen_args = PIPELINE_ARGS(temperature = temperature, top_p = top_p, top_k=top_k, # top_k = 0 then ignore
                        alpha_frequency = alpha_frequency,
                        alpha_presence = alpha_presence,
                        alpha_decay = alpha_decay, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 512)
        cat_char = '🐱'
        bot_char = '🤖'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
                output = generate(self.rwkv,ctx,self.tokenizer,token_count=token_count,args=gen_args,callback=None)
        return output
if __name__ == '__main__':
    base_rwkv_model = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    bi_lora_path = '/media/yueyulin/data_4t/models/lora/biencoder/epoch_1_step_430000/RWKV-x060-World-1B6-v2_rwkv_lora.pth'
    cross_lora_path = '/media/yueyulin/data_4t/models/lora/cross_encoder/epoch_0_step_920000/RWKV-x060-World-1B6-v2_rwkv_lora.pth'
    tokenizer_file = os.path.join(parent_dir,'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    chat_lora_path = '/media/yueyulin/data_4t/models/pissa_r64/epoch_0/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    chat_pissa_path = '/media/yueyulin/data_4t/models/pissa_r64/init_pissa.pth'
    chat_lora_r = 64
    chat_lora_alpha = 64
    fusedEncoder = BiCrossFusionEncoder(
        base_rwkv_model,
        bi_lora_path,
        cross_lora_path,
        chat_lora_path,
        tokenizer,
        chat_lora_r=chat_lora_r,
        chat_lora_alpha=chat_lora_alpha,
        chat_pissa_path=chat_pissa_path)

    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = fusedEncoder.encode_texts(texts)
    outputs = torch.tensor(outputs)
    print(outputs)
    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')
    

    texts_a = ['FAQ是什么？','FAQ是什么？','FAQ是什么？','FAQ是什么？']
    texts_b = ['下图是百度百科对FAQ的解释，我们可以简单的理解其为，网站中的常见问题帮助中心。采用一问一答的方式帮助客户快速解决产品/服务问题！','：FAQ(Frequently Asked Questions)问答系统是目前应用最广泛的问答系统。这种问答系统的结构框架明了、实现简单、容易理解，非常适合作为问答系统入门学习时的观察对象。这里基于本人在问答系统建设方面的“多年”经验，对FAQ问答相关的定义、系统结构、数据集建设、关键技术、应用等方面进行了整理和介绍。','FAQ是英文 Frequently Asked Questions的缩写。中文意思是“常见问题”，或者更通俗点说，“常见问题解答”。FAQ是目前互联网上提供在线帮助的主要方式，通过事先组织一些常见的问答，在网页上发布咨询服务。','从技术，即实现方式的角度来看，问答系统有很多种，包括基于FAQ的问答、基于知识图谱的问答、基于文本的问答等等。这里围绕应用最为广泛的FAQ问答系统，对问答系统的定义、思想、基本结构、方法和应用价值进行介绍。']
    outputs = fusedEncoder.cross_encode_texts(texts_a,texts_b)
    print(outputs)


    instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
    input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问。'
    output = fusedEncoder.sampling_generate(instruction,input_text)
    print(output)

    beam_results = fusedEncoder.beam_generate(instruction,input_text)
    for result in beam_results:
        print(result)