import torch
from torch import nn
from torch._lowrank import svd_lowrank
from torch.nn import functional as F

class LoraEmbedding(nn.Module):
    def __init__(self, 
                 base_layer :nn.Embedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.adapters = set()
        self.active_adapter = "default"
        self.in_features = base_layer.num_embeddings
        self.out_features = base_layer.embedding_dim

    def add_adapter(self, adapter_name: str, r: int, alpha: int) -> None:
        self.adapters.add(adapter_name)
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha/r
        weight_A = torch.randn((r, self.in_features),device=self.base_layer.weight.device)
        weight_B = torch.randn((self.out_features, r),device=self.base_layer.weight.device)
        self.lora_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_B[adapter_name] = nn.Parameter(weight_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter == "default" or self.active_adapter not in self.adapters \
            or self.active_adapter not in self.lora_A \
                or self.active_adapter not in self.lora_B:
            return self.base_layer(x)
        else:
            result = self.base_layer(x)
            embedding_A = self.lora_A[self.active_adapter].T
            embedding_B = self.lora_B[self.active_adapter].T
            scaling = self.scaling[self.active_adapter]
            after_A = F.embedding(x,embedding_A)
            result = result + (after_A @ embedding_B) * scaling
            return result

class LoraLinear(nn.Module):

    def __init__(self, 
                 base_layer: nn.Linear
                 ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.adapters = set()
        self.active_adapter = "default"

    def add_adapter(self, adapter_name: str, r: int, alpha: int) -> None:
        self.adapters.add(adapter_name)
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha/r
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r,bias=False,device=self.base_layer.weight.device)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features,bias=False,device=self.base_layer.weight.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter == "default" or self.active_adapter not in self.adapters \
            or self.active_adapter not in self.lora_A \
                or self.active_adapter not in self.lora_B:
            return self.base_layer(x)
        else:
            A = self.lora_A[self.active_adapter](x)
            B = self.lora_B[self.active_adapter](A)
            return B * self.scaling[self.active_adapter] + self.base_layer(x)

def get_parent_target_module(model: torch.nn.Module, key: str):
    final_dot_index = key.rfind('.')
    if final_dot_index == -1:
        parent_key = None
        target_key = key
    else:
        parent_key = key[:final_dot_index]
        target_key = key[final_dot_index+1:]
    parent_module = model.get_submodule(parent_key) if parent_key is not None else model
    return parent_module, target_key

def inject_lora_adapter_with_state_dict(model: torch.nn.Module, 
                                        adapter_name: str, 
                                        state_dict: dict, 
                                        r:int,
                                        alpha:int,
                                        targets,
                                        peft_name :str = None,
                                        parent_model_name :str = None) -> torch.nn.Module:
    sub_modules = model.named_modules()
    def is_target(key):
        for target in targets:
            if target in key:
                return True
        return False
    for key,sub_module in sub_modules:
        if is_target(key):
            parent_module, target_key = get_parent_target_module(model, key)
            if peft_name is None:
                lora_A = f'{key}.lora_A'
                lora_B = f'{key}.lora_B'
            else:
                if parent_model_name is not None:
                    lora_A = f'{parent_model_name}.{key}.lora_A.{peft_name}'
                    lora_B = f'{parent_model_name}.{key}.lora_B.{peft_name}'
                else:
                    lora_A = f'{key}.lora_A.{peft_name}'
                    lora_B = f'{key}.lora_B.{peft_name}'
                if key != 'emb':
                    lora_A = lora_A+'.weight'
                    lora_B = lora_B+'.weight'
            if lora_A in state_dict and lora_B in state_dict:
                if isinstance(sub_module,nn.Linear):
                        lora_linear = LoraLinear(sub_module)
                        lora_linear.add_adapter(adapter_name,r,alpha)
                        #remove the original module
                        
                        setattr(parent_module,target_key,lora_linear)
                        #find the corresponding lora_A and lora_B
                        lora_linear.lora_A[adapter_name].weight.data = state_dict[lora_A].to(lora_linear.lora_A[adapter_name].weight.device)
                        lora_linear.lora_B[adapter_name].weight.data = state_dict[lora_B].to(lora_linear.lora_B[adapter_name].weight.device)
                elif isinstance(sub_module,LoraLinear):
                    sub_module.add_adapter(adapter_name,r,alpha)
                    sub_module.lora_A[adapter_name].weight.data = state_dict[lora_A].to(sub_module.lora_A[adapter_name].weight.device)
                    sub_module.lora_B[adapter_name].weight.data = state_dict[lora_B].to(sub_module.lora_B[adapter_name].weight.device)
                elif isinstance(sub_module,nn.Embedding):
                    lora_embedding = LoraEmbedding(sub_module)
                    lora_embedding.add_adapter(adapter_name,r,alpha)
                    setattr(parent_module,target_key,lora_embedding)
                    lora_embedding.lora_A[adapter_name].data = state_dict[lora_A].to(lora_embedding.lora_A[adapter_name].data.device)
                    lora_embedding.lora_B[adapter_name].data = state_dict[lora_B].to(lora_embedding.lora_B[adapter_name].data.device)
                elif isinstance(sub_module,LoraEmbedding):
                    sub_module.add_adapter(adapter_name,r,alpha)
                    sub_module.lora_A[adapter_name].data = state_dict[lora_A].to(sub_module.lora_A[adapter_name].data.device)
                    sub_module.lora_B[adapter_name].data = state_dict[lora_B].to(sub_module.lora_B[adapter_name].data.device)

def set_adapter(model, adapter_name):
    sub_modules = model.named_modules()
    for key,sub_module in sub_modules:
        if isinstance(sub_module,LoraLinear) or isinstance(sub_module,LoraEmbedding):
            sub_module.active_adapter = adapter_name



def load_base_model():
    ckpt = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    
    import torch
    device = 'cuda'
    dtype = torch.bfloat16
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(ckpt, args)
    print(args)
    model = RWKV(args)
    info = model.load_state_dict(w)
    model.eval()
    print(info)
    return model

if __name__ == '__main__':
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
    from src.model_run import RWKV,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate

    base_layer = nn.Linear(10, 10)
    lora_linear = LoraLinear(base_layer)
    lora_linear.add_adapter("sft_lora", 8, 32)
    lora_linear.active_adapter = "sft_lora"
    x = torch.randn(10, 10)
    print(lora_linear(x).shape)
    lora_linear.active_adapter = "default"
    print(lora_linear(x).shape)
    lora_linear.active_adapter = 'sfwef'
    print(lora_linear(x).shape)
    test_mode = sys.argv[1]    
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    should_delete_head = True
    if test_mode == 'test_generate':
        test_lora_generate = True
        test_biencoder = False
    elif test_mode == 'test_biencoder':
        test_lora_generate = False
        test_biencoder = True
    elif test_mode == 'test_both':
        test_lora_generate = True
        test_biencoder = True
        should_delete_head = False
    def test_generate(model):
        gen_args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.96, top_k = 20, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 512)
        cat_char = '🐱'
        bot_char = '🤖'
        instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
        input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问。'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        print(ctx)

        states_value = None
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type='cuda',dtype=torch.bfloat16):
                output = generate(model, ctx,tokenizer, token_count=128, args=gen_args,callback=None,state=states_value)
            print(output)

    model = load_base_model()
    device = "cuda"
    dtype = torch.bfloat16
    model = model.to(device=device,dtype=dtype)
    lora_aplha = 32
    lora_r = 8
    if test_lora_generate:    
        test_generate(model)
        lora_ckpt = '/media/yueyulin/data_4t/models/lora_rwkv/epoch_7/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
        lora_state_dict = torch.load(lora_ckpt,map_location='cpu')
        lora_aplha = 32
        lora_r = 8
        targets = {'ffn.key','ffn.value','ffn.receptance','att.key','att.value','att.receptance'}
        inject_lora_adapter_with_state_dict(model,
                                            "sft_lora",
                                            lora_state_dict,
                                            lora_r,
                                            lora_aplha,
                                            targets)
        
        set_adapter(model,"sft_lora")

        test_generate(model)
    if test_biencoder:
        lora_ckpt = '/media/yueyulin/data_4t/models/lora/biencoder/epoch_1_step_430000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
        lora_state_dict = torch.load(lora_ckpt,map_location='cpu')
        #replace lora_embedding_A to lora_A, lora_embedding_B to lora_B
        keys = list(lora_state_dict.keys())
        for key in keys:
            if 'lora_embedding_A' in key:
                new_key = key.replace('lora_embedding_A','lora_A')
                lora_state_dict[new_key] = lora_state_dict.pop(key)
            elif 'lora_embedding_B' in key:
                new_key = key.replace('lora_embedding_B','lora_B')
                lora_state_dict[new_key] = lora_state_dict.pop(key)

        peft_name = 'embedding_lora'
        parent_model_name = "rwkvModel"
        targets = ['emb','ffn.key','ffn.value','ffn.receptance']
        inject_lora_adapter_with_state_dict(model,
                                            "embedding_lora",
                                            lora_state_dict,
                                            lora_r,
                                            lora_aplha,
                                            targets,
                                            peft_name,
                                            parent_model_name)
        from src.model_run import RwkvForSequenceEmbedding
        add_mlp = 'dense.weight' in lora_state_dict
        output_dim = -1
        if add_mlp:
            output_dim = lora_state_dict['dense.weight'].shape[0]
            print(f'add_mlp: {add_mlp},with output_dim: {output_dim}')
        rwkv_embedding = RwkvForSequenceEmbedding(model,
                                                     add_mlp=add_mlp,
                                                     output_dim=output_dim,
                                                     should_delete_head=should_delete_head)
        rwkv_embedding.dense.weight.data = lora_state_dict['dense.weight'].to(device=device,dtype=dtype)
        rwkv_embedding.dense.bias.data = lora_state_dict['dense.bias'].to(device=device,dtype=dtype)
        # rwkv_embedding = rwkv_embedding.to(device=device,dtype=dtype)
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
        set_adapter(model,"embedding_lora")
        outputs = [encode_texts(text) for text in texts]
        print(outputs)
        from sentence_transformers.util import pairwise_cos_sim
        for qid in range(len(texts)):
            query = outputs[qid]
            for i in range(len(texts)):
                if i != qid:
                    print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

            print('-----------------------')

    if test_lora_generate:
        set_adapter(model,"sft_lora")

        test_generate(model)

    if test_biencoder:
        set_adapter(model,"embedding_lora")
        outputs = [encode_texts(text) for text in texts]
        print(outputs)
        from sentence_transformers.util import pairwise_cos_sim
        for qid in range(len(texts)):
            query = outputs[qid]
            for i in range(len(texts)):
                if i != qid:
                    print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

            print('-----------------------')