import os
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'
from rwkv.model import RWKV 
import torch
from rwkv.utils import PIPELINE_ARGS,PIPELINE
class StatesRunner:
    def __init__(self, model_file,strategy,device,dtype):
        self.model = RWKV(model_file,strategy=strategy)
        self.strategy = strategy
        self.device = device
        self.dtype = dtype
        self.states = {}
        self.default_gen_args = PIPELINE_ARGS(
            temperature = 1, top_p = 0.96, top_k = 20, 
            alpha_frequency = 0.25, 
            alpha_presence = 0.25,
            alpha_decay = 0.996, 
            token_ban = [], 
            token_stop = [0,1], 
            chunk_len = 256)
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        
    def add_states(self,states_name,states_file):
        states = torch.load(states_file)
        self.states[states_name] = states
        
    def generate(self,ctx,token_count=512,gen_args=None,states_name=None):
        if states_name is None or states_name not in self.states:
            states_value = None
        else:
            states = self.states[states_name]
            args = self.model.args
            states_value = []
            for i in range(args.n_layer):
                key = f'blocks.{i}.att.time_state'
                value = states[key]
                prev_x = torch.zeros(args.n_embd,device=self.device,dtype=self.dtype)
                prev_states = torch.tensor(value,device=self.device,dtype=self.dtype).transpose(1,2)
                prev_ffn = torch.zeros(args.n_embd,device=self.device,dtype=self.dtype)
                states_value.append(prev_x)
                states_value.append(prev_states)
                states_value.append(prev_ffn)
        if gen_args is None:
            gen_args = self.default_gen_args
            
        return self.pipeline.generate(ctx,token_count=token_count,args=gen_args,state=states_value)
    

if __name__ == '__main__':
    model_path = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'
    domain_state_file = '/media/yueyulin/data_4t/models/states_tuning/4personal_domain/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    statesRunner = StatesRunner(model_path,'cuda:0 fp16','cuda:0',torch.half)
    statesRunner.add_states('domain',domain_state_file)
    
    cat_char = '🐱'
    bot_char = '🤖'
    instruction ='根据input中文本内容，协助用户识别文本所属的领域。随后，找出与该领域关联最紧密的专家。接着，作为输出，列举出五至十项可在该文本中执行的具体任务。接下来，提取以下信息：领域：对于给定的示例文本，帮助用户指定一个描述性领域，概括文本的主题。请按照JSON字符串的格式回答，无法提取则不输出'
    input_text = '{\"input\":\"超长期特别国债（ultra-long special treasury bonds），一般指发行期限在10年以上的，为特定目标发行的、具有明确用途的国债。超长期特别国债专项用于国家重大战略实施和重点领域安全能力建设，2024年先发行1万亿元，期限分别为20年、30年、50年。 [1]\
    2024年5月13日，财政部网站公布2024年一般国债、超长期特别国债发行有关安排。 [6-7]2024年5月17日，30年期超长期特别国债正式首发。根据发行安排，首发的30年期超长期特别国债，为固定利率附息债，总额400亿元。 [8]6月14日,财政部发行2024年超长期特别国债（三期）（50年期），竞争性招标面值总额350亿元。 [13]7月24日，通过财政部政府债券发行系统招标发行550亿元30年期超长期特别国债，票面利率在当天通过竞争性招标确定。 [15]\"}'
    ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
    print(ctx)
    print('start to generate...')
    output = statesRunner.generate(ctx,token_count=512,states_name='domain')
    print(output)