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
from src.model_ext import RwkvForSequenceEmbedding,load_ckpt_and_parse_args,create_empty_args,RwkvForClassification
from src.model import RWKV
import torch
from torch.cuda import amp
from peft.tuners.lora.layer import LoraLayer

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
                 tokenizer,
                 lora_type='lora',
                 lora_r=8,
                 lora_alpha=32,
                 target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                 cross_adapter_name='cross_encoder_lora',
                 original_cross_adapter_name='embedding_lora',
                 bi_adapter_name='bi_embedding_lora',
                 original_bi_adapter_name='embedding_lora',
                 sep_token_id = 2
                 ) -> None:
        self.base_rwkv = base_rwkv
        args = create_empty_args()
        w = load_ckpt_and_parse_args(base_rwkv,args)
        rwkv = RWKV(args)
        info = rwkv.load_state_dict(w)
        print(f'load model from {base_rwkv},result is {info}')
        #load cross encoder and inject cross adapter
        cross_lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            cross_lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            cross_lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        from peft import inject_adapter_in_model
        rwkv = inject_adapter_in_model(cross_lora_config,rwkv,adapter_name=cross_adapter_name)

        #load bi encoder and inject bi adapter
        bi_lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            bi_lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            bi_lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        rwkv = inject_adapter_in_model(bi_lora_config,rwkv,adapter_name=bi_adapter_name)
        self.cross_encoder = RwkvForClassification(rwkv)
        self.bi_encoder = RwkvForSequenceEmbedding(rwkv,add_mlp=True,output_dim=1024)

        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id

        #load cross encoder lora params
        with torch.no_grad():
            w = torch.load(cross_lora_path)
            #replace keys with original adapter name to new adapter name
            if original_cross_adapter_name != cross_adapter_name:
                print(f'origal_keys:{list(w.keys())}')
                for k in list(w.keys()):
                    if original_cross_adapter_name in k:
                        new_k = k.replace(original_cross_adapter_name,cross_adapter_name)
                        w[new_k] = w.pop(k)
            info = self.cross_encoder.load_state_dict(w,strict=False)
            print(f'load model from {cross_lora_path},result is {info}')
        self.cross_encoder = self.cross_encoder.bfloat16()
        self.cross_encoder = self.cross_encoder.cuda()
        self.cross_encoder.eval()

        #load bi encoder lora params
        with torch.no_grad():
            w = torch.load(bi_lora_path)
            #replace keys with original adapter name to new adapter name
            if original_bi_adapter_name != bi_adapter_name:
                print(f'origal_keys:{list(w.keys())}')
                for k in list(w.keys()):
                    if original_bi_adapter_name in k:
                        new_k = k.replace(original_bi_adapter_name,bi_adapter_name)
                        w[new_k] = w.pop(k)
            info = self.bi_encoder.load_state_dict(w,strict=False)
            print(f'load model from {bi_lora_path},result is {info}')
        self.bi_encoder = self.bi_encoder.bfloat16()
        self.bi_encoder = self.bi_encoder.cuda()
        self.bi_encoder.eval()

        self.bi_adapter_name = bi_adapter_name
        self.cross_adapter_name = cross_adapter_name
        self.rwkv = rwkv

        print(f'inject lora from {cross_lora_path} and {bi_lora_path} to model,result is {rwkv}')


    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.rwkv.modules():
            if isinstance(module, LoraLayer):
                module.set_adapter(adapter_name)

    def encode_texts(self,texts):
        self.set_adapter(self.bi_adapter_name)
        MAX_LEN = 4096
        max_len = 0
        input_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)[0:MAX_LEN-1]
            ids.append(self.bi_encoder.embedding_id)
            max_len = max(max_len,len(ids))
            input_ids.append(ids)
        input_ids = [ids + [self.bi_encoder.pad_id]*(max_len-len(ids)) for ids in input_ids]
        input_ids = torch.tensor(input_ids,dtype=torch.long,device='cuda')
        with torch.no_grad():
            with amp.autocast_mode.autocast(enabled=True,dtype=torch.bfloat16):
                outputs = self.bi_encoder.forward(input_ids)
        return outputs
    
    def cross_encode_texts(self,texts_a, texts_b):
        assert len(texts_a) == len(texts_b)
        self.set_adapter(self.cross_adapter_name)
        MAX_LEN = 4096
        max_len = 0
        texts_a_ids = [self.tokenizer.encode(text) for text in texts_a]
        texts_b_ids = [self.tokenizer.encode(text) for text in texts_b]
        all_input_ids = []
        for i in range(len(texts_a_ids)):
            input_ids = texts_a_ids[i]+[self.sep_token_id]+texts_b_ids[i]+[self.cross_encoder.class_id][0:MAX_LEN]
            if len(input_ids) == MAX_LEN:
                #last token_id has to be class_id
                input_ids[-1] = self.cross_encoder.class_id
            max_len = max(max_len,len(input_ids))
            all_input_ids.append(input_ids)
        all_input_ids = [ids + [self.cross_encoder.pad_id]*(max_len-len(ids)) for ids in all_input_ids]
        with torch.no_grad():
            with amp.autocast_mode.autocast(enabled=True,dtype=torch.bfloat16):
                all_input_ids = torch.tensor(all_input_ids,dtype=torch.long,device='cuda')
                outputs = self.cross_encoder.forward(all_input_ids)
        return outputs

class CrossEncoder:
    def __init__(self,
                    base_rwkv,
                    lora_path,
                    tokenizer,
                    lora_type='lora',
                    lora_r=8,
                    lora_alpha=32,
                    target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                    adapter_name='cross_encoder_lora',
                    original_adapter_name='embedding_lora',
                    sep_token_id = 2) -> None:
        self.base_rwkv = base_rwkv
        args = create_empty_args()
        w = load_ckpt_and_parse_args(base_rwkv,args)
        rwkv = RWKV(args)
        info = rwkv.load_state_dict(w)
        print(f'load model from {base_rwkv},result is {info}')
        self.lora_path = lora_path
        lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        from peft import inject_adapter_in_model
        rwkv = inject_adapter_in_model(lora_config,rwkv,adapter_name=adapter_name)
        print(f'inject lora from {lora_path} to model,result is {rwkv}')
        self.adapter_name = adapter_name
        self.cross_encoder = RwkvForClassification(rwkv)
        with torch.no_grad():
            w = torch.load(lora_path)
            #replace keys with original adapter name to new adapter name
            if original_adapter_name != adapter_name:
                print(f'origal_keys:{list(w.keys())}')
                for k in list(w.keys()):
                    if original_adapter_name in k:
                        new_k = k.replace(original_adapter_name,adapter_name)
                        w[new_k] = w.pop(k)
            info = self.cross_encoder.load_state_dict(w,strict=False)
            print(f'load model from {lora_path},result is {info}')
        self.cross_encoder = self.cross_encoder.bfloat16()
        self.cross_encoder = self.cross_encoder.cuda()
        self.cross_encoder.eval()
        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id

    def encode_texts(self,texts_a, texts_b):
        assert len(texts_a) == len(texts_b)
        MAX_LEN = 4096
        max_len = 0
        texts_a_ids = [self.tokenizer.encode(text) for text in texts_a]
        texts_b_ids = [self.tokenizer.encode(text) for text in texts_b]
        all_input_ids = []
        for i in range(len(texts_a_ids)):
            input_ids = texts_a_ids[i]+[self.sep_token_id]+texts_b_ids[i]+[self.cross_encoder.class_id][0:MAX_LEN]
            if len(input_ids) == MAX_LEN:
                #last token_id has to be class_id
                input_ids[-1] = self.cross_encoder.class_id
            max_len = max(max_len,len(input_ids))
            all_input_ids.append(input_ids)
        all_input_ids = [ids + [self.cross_encoder.pad_id]*(max_len-len(ids)) for ids in all_input_ids]
        with torch.no_grad():
            with amp.autocast_mode.autocast(enabled=True,dtype=torch.bfloat16):
                all_input_ids = torch.tensor(all_input_ids,dtype=torch.long,device='cuda')
                outputs = self.cross_encoder.forward(all_input_ids)
        return outputs
    
class BiEncoder:
    def __init__(self,
                    base_rwkv,
                    lora_path,
                    tokenizer,
                    lora_type='lora',
                    add_mlp=True,
                    mlp_dim=1024,
                    lora_r=8,
                    lora_alpha=32,
                    target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                    adapter_name='bi_embedding_lora',
                    original_adapter_name='embedding_lora') -> None:
        self.base_rwkv = base_rwkv
        self.add_mlp = add_mlp
        self.mlp_dim = mlp_dim
        args = create_empty_args()
        w = load_ckpt_and_parse_args(base_rwkv,args)
        rwkv = RWKV(args)
        info = rwkv.load_state_dict(w)
        print(f'load model from {base_rwkv},result is {info}')

        self.lora_path = lora_path
        lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        from peft import inject_adapter_in_model
        rwkv = inject_adapter_in_model(lora_config,rwkv,adapter_name=adapter_name)
        self.adapter_name = adapter_name
        print(f'inject lora from {lora_path} to model,result is {rwkv}')

        self.rwkv_embedding = RwkvForSequenceEmbedding(rwkv,add_mlp=add_mlp,output_dim=mlp_dim)
        print(self.rwkv_embedding)
        with torch.no_grad():
            w = torch.load(lora_path)
            #replace keys with original adapter name to new adapter name
            if original_adapter_name != adapter_name:
                print(f'origal_keys:{list(w.keys())}')
                for k in list(w.keys()):
                    if original_adapter_name in k:
                        new_k = k.replace(original_adapter_name,adapter_name)
                        w[new_k] = w.pop(k)
            info = self.rwkv_embedding.load_state_dict(w,strict=False)
            print(f'load model from {lora_path},result is {info}')
        self.rwkv_embedding = self.rwkv_embedding.bfloat16()
        self.rwkv_embedding = self.rwkv_embedding.cuda()
        self.rwkv_embedding.eval()
        self.tokenizer = tokenizer

    def encode_texts(self,texts):
        MAX_LEN = 4096
        max_len = 0
        input_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)[0:MAX_LEN-1]
            ids.append(self.rwkv_embedding.embedding_id)
            max_len = max(max_len,len(ids))
            input_ids.append(ids)
        input_ids = [ids + [self.rwkv_embedding.pad_id]*(max_len-len(ids)) for ids in input_ids]
        input_ids = torch.tensor(input_ids,dtype=torch.long,device='cuda')
        with torch.no_grad():
            with amp.autocast_mode.autocast(enabled=True,dtype=torch.bfloat16):
                outputs = self.rwkv_embedding.forward(input_ids)
        return outputs
if __name__ == '__main__':
    base_rwkv_model = '/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/bi-encoder/add_mlp_in_batch_neg/epoch_0_step_200000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    tokenizer_file = os.path.join(parent_dir,'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    bi_encoder = BiEncoder(base_rwkv_model,lora_path,tokenizer)
    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = bi_encoder.encode_texts(texts)
    print(outputs)
    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')
    
    del bi_encoder

    cross_lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/cross-encoder/epoch_0_step_500000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    cross_encoder = CrossEncoder(base_rwkv_model,cross_lora_path,tokenizer)
    print(cross_encoder.cross_encoder)
    texts_a = ['我打算取消订单','我打算取消订单','我打算取消订单','我打算取消订单']
    texts_b = ['订单在什么情况可以取消？','我要退款','国家主席习近平在人民大会堂同来华进行国事访问的厄立特里亚总统伊萨亚斯举行会谈。 5月15日下午，国家主席习近平在北京人民大会堂同来华进行国事访问的厄立特里亚总统伊萨亚斯举行会谈。新华社记者 鞠鹏 摄 习近平指出，中国和厄立特里亚传统友谊深厚。今年5月24日，两国将迎来建交30周年纪念日。30年来，中厄始终相互信任、相互支持。中方从战略高度和长远角度看待和发展中厄关系，是厄方可信赖的朋友。面对当前充满不稳定性、不确定性的国际形势，发展好中厄关系不仅符合两国共同和长远利益，也对维护地区和平和国际公平正义具有重要意义。中方愿同厄方深化两国朋友加同志的亲密友好关系，推动中厄战略伙伴关系不断迈上新台阶。','生活喜忧参半，人生苦乐酸甜，人活一辈子没有我们想象的如此美好的生活，但是日子也没有我们想的那么糟糕。世界上最多的是平凡人，没有翻云覆雨的能力，也没有只手遮天的权利，唯一有的就是过好自己的生活，日子比上不足比下有余，不羡慕比自己生活好的人，也不低看不如自己的人。 人生的轨迹不会一尘不变，有站在低谷的时候，就有矗立山巅的时候，不要害怕狂风暴雨，因为风雨过后才会出现美丽的彩虹；没有白受的罪，没有谁凭空享福，好的生活都是一点一滴奋斗出来的，生活没有那么完美，但只要努力就不会糟糕。 这个世界上总些人受点伤就哭天喊地，也有些人咬牙低头默默努力；如果现在的生活不美好，那可能是因为你不够努力；生活就是一场马拉松，坚持到底的人才是胜者；生活原本就是一件很难的事情，想要过上自己喜欢的生活更是难上加难！ 生活喜忧参半，人生苦乐酸甜，人活一辈子没有我们想象的如此美好的生活，但是日子也没有我们想的那么糟糕。世界上最多的是平凡人，没有翻云覆雨的能力，也没有只手遮天的权利，唯一有的就是过好自己的生活，日子比上不足比下有余，不羡慕比自己生活好的人，也不低看不如自己的人。']
    outputs = cross_encoder.encode_texts(texts_a,texts_b)
    print(outputs)

    del cross_encoder

    fused_encoder = BiCrossFusionEncoder(base_rwkv_model,lora_path,cross_lora_path,tokenizer)
    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = fused_encoder.encode_texts(texts)
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')
    

    outputs = fused_encoder.cross_encode_texts(texts_a,texts_b)
    print(outputs)