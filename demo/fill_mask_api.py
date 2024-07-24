from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import time
import torch
import argparse
from transformers import AutoTokenizer
model = None
tokenizer = None
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
    os.environ['WKV'] = ''
    os.environ['RWKV_TRAIN_TYPE'] = ''
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    os.environ['NO_CUDA'] = '1'
setup_env()
from src.model_encoder_run import RwkvEncoder
from src.model_run import create_empty_args,load_embedding_ckpt_and_parse_args
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
    print(mask_positions)
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
                        results.append({'output_text':tokenizer.decode(texts_idx[b]),'score':prob})
                        if cum_prob > MAX_CUM_PROB:
                            break
                    mask_idx += 1
    return results
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Test MLM model")
    parser.add_argument("--model_file",type=str,default='/media/yueyulin/data_4t/models/mlm/final/epoch_0_step_100000/RWKV-x060-MLM-ctx4096.pth.pth')
    args = parser.parse_args() 
    print(args)
    model = load_base_model(args.model_file)
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)