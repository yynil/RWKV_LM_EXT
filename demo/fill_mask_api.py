from fastapi import FastAPI, Request,HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import torch
import argparse
import json
from sentence_transformers.util import cos_sim as sim_fn
model = None
emb_model = None
tokenizer = None
states_runner = None
states_configuration = None
def setup_env():
    import os
    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    import sys
    sys.path.append(parent_path)
    print(f'add path: {parent_path} to sys.path')
    os.environ['RWKV_JIT_ON'] = '1'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    os.environ['WKV'] = ''
    os.environ['RWKV_TRAIN_TYPE'] = ''
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    os.environ['NO_CUDA'] = '1'
setup_env()

from transformers import AutoTokenizer
from src.model_encoder_run import RwkvEncoder
from src.model_run import create_empty_args,load_embedding_ckpt_and_parse_args
from infer.rwkv_states_runner import StatesRunner
def load_base_model(base_model):
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(base_model, args)
    print(args)
    args.emb_id = 151329
    args.pad_id = 151334
    args.mask_id = 151330
    model = RwkvEncoder(args)
    info = model.load_state_dict(w)
    print(info)
    return model
app = FastAPI()

class MaskFillResult(BaseModel):
    output_text: str
    score: float

class FillMaskResponse(BaseModel):
    results: Optional[List[MaskFillResult]] = None
    elapsed_time: str
    msg: Optional[str] = None
# 定义请求和响应模型
class SentenceSimilarityRequest(BaseModel):
    input_text: str
    compared_texts: List[str]

class SentenceSimilarityResponse(BaseModel):
    similarities: Optional[List[float]] = None
    elapsed_time: str
    msg: Optional[str] = None

class InputData(BaseModel):
    input_text: str
    states_name: str

class OutputData(BaseModel):
    output_text: str
    elapsed_time: float

@app.post("/process_text", response_model=OutputData)
def process_text(data: InputData):
    start_time = time.time()
    
    # 假设states_runner有一个process方法来处理输入文本
    try:
        instruction = ''
        if data.states_name in states_configuration:
            instruction = states_configuration[data.states_name]['instruction']
        cat_char = '🐱'
        bot_char = '🤖'
        input_data = {'input':data.input_text}
        ctx = f'{cat_char}:{instruction}\n{json.dumps(input_data,ensure_ascii=False)}\n{bot_char}:'
        output_text = states_runner.generate(ctx,token_count=512,states_name=data.states_name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    elapsed_time = time.time() - start_time
    return OutputData(output_text=output_text, elapsed_time=elapsed_time)

def load_states_config(states_file_config):
    with open(states_file_config,'r') as f:
        return json.load(f)



# 添加新的路由处理函数
@app.post("/compute_sentence_similarities")
async def compute_sentence_similarities(request: Request, similarity_request: SentenceSimilarityRequest):
    # 记录开始时间
    start_time = time.time()

    # 检查 input_text 是否为字符串
    if not isinstance(similarity_request.input_text, str):
        end_time = time.time()
        elapsed_time = f"{end_time - start_time:.4f}"
        return SentenceSimilarityResponse(
            similarities=None,
            elapsed_time=elapsed_time,
            msg="input_text 必须是字符串"
        )

    # 检查 compared_texts 是否为字符串数组
    if not isinstance(similarity_request.compared_texts, list) or not all(isinstance(text, str) for text in similarity_request.compared_texts):
        end_time = time.time()
        elapsed_time = f"{end_time - start_time:.4f}"
        return SentenceSimilarityResponse(
            similarities=None,
            elapsed_time=elapsed_time,
            msg="compared_texts 必须是字符串数组"
        )

    # 计算相似度（此处为示例，实际计算逻辑需要根据具体需求实现）
    scores, elapsed_time = compute_similarities_internal([similarity_request.input_text] + similarity_request.compared_texts)

    return SentenceSimilarityResponse(similarities=scores, elapsed_time=elapsed_time)

@app.post("/fill_mask")
async def fill_mask(request: Request, input_text: str):
    # 记录开始时间
    start_time = time.time()

    # 记录请求的IP
    client_ip = request.client.host
    print(f"Received request from IP: {client_ip}")

    # 检查 input_text
    if len(input_text) > 128:
        end_time = time.time()
        elapsed_time = f"{end_time - start_time:.4f}"
        return FillMaskResponse(
            results=None,
            elapsed_time=elapsed_time,
            msg="输入文本长度不能超过128个字符"
        )

    mask_count = input_text.count("[MASK]")
    if mask_count != 1:
        end_time = time.time()
        elapsed_time = f"{end_time - start_time:.4f}"
        return FillMaskResponse(
            results=None,
            elapsed_time=elapsed_time,
            msg="输入文本必须包含且仅包含一个[MASK]标记"
        )


    # 调用 fill_mask_internal 处理合法输入
    internal_results = fill_mask_internal(input_text)
    # 将内部处理结果转换为 MaskFillResult 对象列表
    results = [MaskFillResult(output_text=result['output_text'], score=result['score']) for result in internal_results]
    
    # 计算执行时间
    end_time = time.time()
    elapsed_time = f"{end_time - start_time:.4f}"
    
    return FillMaskResponse(results=results, elapsed_time=elapsed_time)

def fill_mask_internal(input_text):
    emb_id = 151329
    pad_id = 151334
    mask_id = 151330
    texts_idx = tokenizer.encode(input_text,add_special_tokens=False)
    texts_idx.append(emb_id)
    texts_idx = [texts_idx]
    mask_positions = []
    for text_idx in texts_idx:
        mask_positions.append([i for i, x in enumerate(text_idx) if x == args.mask_id])
    device = 'cpu'
    input_ids = torch.tensor(texts_idx,dtype=torch.long,device=device)
    MAX_CUM_PROB = 0.7
    import time
    results = []
    with torch.no_grad():
        with torch.autocast(device_type=device,dtype=torch.float32):
            logits = model.forward(input_ids)
            for b in range(len(texts_idx)):
                mask_position = mask_positions[b]
                masked_prob = torch.softmax(logits[b,mask_position],dim=-1)
                probs,indices = torch.topk(masked_prob,10)
                for position in mask_position:
                    cum_prob = 0
                    mask_idx = 0
                    for i in range(10):
                        texts_idx[b][position] = indices[mask_idx][i].item()
                        prob = probs[mask_idx][i].item()
                        cum_prob += prob
                        results.append({'output_text':tokenizer.decode(texts_idx[b],skip_special_tokens=True),'score':prob})
                        if cum_prob > MAX_CUM_PROB:
                            break
                    mask_idx += 1
    return results

def compute_similarities_internal(texts :List[str],device:str='cpu'):
    emb_id = 151329
    pad_id = 151334
    MAX_LEN = 4096
    texts_idx = [tokenizer.encode(text,add_special_tokens=False) for text in texts]
    texts_idx = [ text_idx[:MAX_LEN-1] + [emb_id] for text_idx in texts_idx]
    max_len = max([len(text_idx) for text_idx in texts_idx])
    texts_idx = [text_idx + [pad_id]*(max_len-len(text_idx)) for text_idx in texts_idx]
    input_ids = torch.tensor(texts_idx,dtype=torch.long,device=device)
    import time
    with torch.no_grad():
        with torch.autocast(device_type=device,dtype=torch.float32):
            start_time = time.time()
            embs = emb_model.encode_sentence(input_ids)
            end_time = time.time()
            scores = sim_fn(embs[0],embs[1:])
            return scores.squeeze(0).tolist(),f'{end_time-start_time:.4f}'


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Test MLM/Emb model")
    parser.add_argument("--model_file",type=str,default='/media/yueyulin/data_4t/models/mlm/final/epoch_0_step_100000/RWKV-x060-MLM-ctx4096.pth.pth')
    parser.add_argument("--emb_model_file",type=str,default='/media/yueyulin/KINGSTON/models/all_chinese_biencoder/trainable_model/epoch_9/RWKV-x060-MLM-ctx4096.pth.pth')
    parser.add_argument("--device",type=str,default='cuda:0')
    parser.add_argument('--dtype',type=str,default='fp16',choices=['fp16','fp32','bf16'])
    parser.add_argument('--states_file_config',type=str)
    parser.add_argument('--llm_model_file',type=str,default='/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth')
    args = parser.parse_args() 
    print(args)
    ###Init llm
    strategy = f'{args.device} {args.dtype}'
    if args.dtype == 'fp16':
        dtype = torch.half
    elif args.dtype == 'fp32':
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    states_runner = StatesRunner(args.llm_model_file,strategy,args.device,dtype)
    states_configuration = load_states_config(args.states_file_config)
    for states_name in states_configuration.keys():
        states_file = states_configuration[states_name]['file']
        states_runner.add_states(states_name,states_file)
    cat_char = '🐱'
    bot_char = '🤖'
    instruction ='根据input中文本内容，协助用户识别文本所属的领域。随后，找出与该领域关联最紧密的专家。接着，作为输出，列举出五至十项可在该文本中执行的具体任务。接下来，提取以下信息：领域：对于给定的示例文本，帮助用户指定一个描述性领域，概括文本的主题。请按照JSON字符串的格式回答，无法提取则不输出'
    input_text = '{\"input\":\"超长期特别国债（ultra-long special treasury bonds），一般指发行期限在10年以上的，为特定目标发行的、具有明确用途的国债。超长期特别国债专项用于国家重大战略实施和重点领域安全能力建设，2024年先发行1万亿元，期限分别为20年、30年、50年。 [1]\
    2024年5月13日，财政部网站公布2024年一般国债、超长期特别国债发行有关安排。 [6-7]2024年5月17日，30年期超长期特别国债正式首发。根据发行安排，首发的30年期超长期特别国债，为固定利率附息债，总额400亿元。 [8]6月14日,财政部发行2024年超长期特别国债（三期）（50年期），竞争性招标面值总额350亿元。 [13]7月24日，通过财政部政府债券发行系统招标发行550亿元30年期超长期特别国债，票面利率在当天通过竞争性招标确定。 [15]\"}'
    ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
    print(ctx)
    print('start to generate...')
    output = states_runner.generate(ctx,token_count=512,states_name='domain_expert')
    print(output)
    ###
    model = load_base_model(args.model_file)
    emb_model = load_base_model(args.emb_model_file)
    tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
    args.emb_id = 151329
    args.pad_id = 151334
    args.mask_id = 151330
    dtype = torch.float32
    model = model.to(dtype=dtype)
    print(model)
    input_text = '你是天边的云彩，我是[MASK]的风筝。'
    results = fill_mask_internal(input_text)
    print(results)
    texts = ['每天吃苹果有什么好处？',
             '宁神安眠：苹果中含有的磷和铁等元素，易被肠壁吸收，有补脑养血、宁神安眠作用。苹果的香气是治疗抑郁和压抑感的良药。研究发现，在诸多气味中，苹果的香气对人的心理影响最大，它具有明显的消除心理压抑感的作用。',
             '美白养颜、降低胆固醇：苹果中的胶质和微量元素铬能保持血糖的稳定，还能有效地降低胆固醇。苹果中的粗纤维可促进肠胃蠕功，并富含铁、锌等微量元素，可使皮肤细润有光泽，起到美容瘦身的作用。',
             '苹果生吃治便秘，熟吃治腹泻：苹果中含有丰富的鞣酸、果胶、膳食纤维等特殊物质，鞣酸是肠道收敛剂，它能减少肠道分泌而使大便内水分减少，从而止泻。而果胶则是个“两面派”，未经加热的生果胶有软化大便缓解便秘的作用，煮过的果胶却摇身一变，具有收敛、止泻的功效。膳食纤维又起到通便作用。',
             '保护心脏：苹果的纤维、果胶、抗氧化物等能降低体内坏胆固醇并提高好胆固醇含量，所以每天吃一两个苹果不容易得心脏病。']
    scores,elapsed_time = compute_similarities_internal(texts)
    print(scores)
    print(elapsed_time)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)