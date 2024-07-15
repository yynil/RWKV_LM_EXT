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
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
        

class StatesGenerator:
    def __init__(self,base_model,tokenizer_file,device='cuda'):
        self.base_model = base_model
        args = create_empty_args()
        w = load_embedding_ckpt_and_parse_args(base_model, args)
        model = RWKV(args)
        info = model.load_state_dict(w)
        model.eval()
        self.dtype = torch.bfloat16
        self.device = device
        self.model = model.to(self.dtype).to(self.device)
        del w
        self.states = {}
        self.args = args
        self.tokenizer = TRIE_TOKENIZER(tokenizer_file)

    def load_states(self,states_file,states_name):
        args = self.args
        states = torch.load(states_file)
        states_value = []
        n_head = args.n_head
        head_size = args.head_size_a
        for i in range(args.n_layer):
            key = f'blocks.{i}.att.time_state'
            value = states[key]
            prev_x = torch.zeros(args.n_embd,device=self.device,dtype=torch.float)
            prev_states = torch.tensor(value,device=self.device,dtype=torch.float).transpose(1,2)
            prev_ffn = torch.zeros(args.n_embd,device=self.device,dtype=torch.float)
            states_value.append(prev_x)
            states_value.append(prev_states)
            states_value.append(prev_ffn)
        self.states[states_name] = states_value

    def get_states(self,states_name):
        if states_name not in self.states:
            raise None
        else:
            states_copy = []
            for s in self.states[states_name]:
                states_copy.append(s.clone())
            return states_copy


    def generate(self,input_text,instruction,states_name,temperature = 1.0, top_p = 0.96, top_k = 20, alpha_frequency = 0.25, alpha_presence = 0.25, alpha_decay = 0.996, token_ban = [], token_stop = [0,1], chunk_len = 512,gen_count=128):
        args = self.args
        model = self.model
        states = self.get_states(states_name)
        gen_args = PIPELINE_ARGS(temperature = temperature, top_p = top_p, top_k = top_k, alpha_frequency = alpha_frequency, alpha_presence = alpha_presence, alpha_decay = alpha_decay, token_ban = token_ban, token_stop = token_stop, chunk_len = chunk_len)
        cat_char = '🐱'
        bot_char = '🤖'
        instruction ='你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type='cuda',dtype=self.dtype):
                output = generate(model,ctx,self.tokenizer,token_count=gen_count,args=gen_args,state=states)
        return output
    
if __name__ == '__main__':
    model_file = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'

    kg_states_file = '/media/yueyulin/data_4t/models/states_tuning/instructKGC_scattered/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    type_states_file = '/media/yueyulin/data_4t/models/states_tuning/kg_type/20240702-105004/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    unit_extractor_states_file = '/media/yueyulin/data_4t/models/states_tuning/units_extractor/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    sg = StatesGenerator(model_file,tokenizer_file)
    sg.load_states(kg_states_file,'kg')
    sg.load_states(type_states_file,'type')
    sg.load_states(unit_extractor_states_file,'unit_extractor')
    from kg_schema import whole_schema,all_types
    import json
    kg_instruction = '你是一个图谱实体知识结构化专家。请从input中抽取出符合schema定义的实体实例和其属性，不存在的属性不输出，属性存在多值就返回列表。请按照JSON字符串的格式回答。'
    type_instruction = '请根据input中的文本内容，选择schema中的最符合input内容的类别输出，用JSON格式输出。'
    text = "个人简介姓名：拉塞·维比  所属球队：布伦特福德  国籍：丹麦、法国、荷兰、法属圭亚那  出生日期：1987-02-22  身高：181cm  体重：73kg  场上位置：前锋  球衣号码：21  丹麦射手拉塞-维比，获得了2014赛季瑞超联赛金靴"
    input_text = json.dumps({'input':text,'schema':all_types},ensure_ascii=False)
    print(input_text)
    type_output = sg.generate(input_text,type_instruction,'type',top_k=0,top_p=0,gen_count=10)
    print(type_output)
    # type_output = json.loads(type_output)
    schema_type = '人物'
    print(schema_type)
    schema = whole_schema[schema_type]
    input_text = json.dumps({'input':text,'schema':schema},ensure_ascii=False)
    kg_output = sg.generate(input_text,kg_instruction,'kg',top_k=0,top_p=0,gen_count=2048)
    print(kg_output)
    kg_output = json.loads(kg_output)
    # print(kg_output['result'])

    unit_instruction = '你是一个单位提取专家。请从input中抽取出数字和单位，请按照JSON字符串的格式回答，无法提取则不输出。'
    input_text = '大约503万平方米'
    unit_output = sg.generate(input_text,unit_instruction,'unit_extractor',top_k=0,top_p=0,gen_count=128)
    print(unit_output)
    input_text = '4845人'
    unit_output = sg.generate(input_text,unit_instruction,'unit_extractor',top_k=0,top_p=0,gen_count=128)
    print(unit_output)
    input_text = '约89434户'
    unit_output = sg.generate(input_text,unit_instruction,'unit_extractor',top_k=0,top_p=0,gen_count=128)
    print(unit_output)
    input_text = '可能有38.87亿平方公里'
    unit_output = sg.generate(input_text,unit_instruction,'unit_extractor',top_k=0,top_p=0,gen_count=128)
    print(unit_output)