import torch
import argparse
from transformers import AutoTokenizer
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test MLM model")
    parser.add_argument("--model_file",type=str,default='/media/yueyulin/data_4t/models/mlm/final/epoch_0_step_100000/RWKV-x060-MLM-ctx4096.pth.pth')
    args = parser.parse_args() 
    print(args)
    model = load_base_model(args.model_file)

    args.emb_id = 151329
    args.pad_id = 151334
    args.mask_id = 151330
    print(model)
    device = 'cpu'
    model = model.to(device=device,dtype=torch.float32)
    texts = ['法国的首都在[MASK]。',
             '[MASK]首都在北京。',
             '生活的真谛是[MASK]。',
             '在二战中，阿道夫·希特勒是[MASK]。',
             '1949年十月一号，发生了一件大事，那就是中华人民共和国[MASK]。',
             '原子的行星模型，现在普遍认为是[MASK]的。',
             '根据量子场论的解释，粒子的质量来自[MASK]的作用。',
             '雨后，彩虹出现在天边，小美陶醉地说："真[MASK]啊!"',]
    tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
    print(tokenizer(texts,padding=True,add_special_tokens=False))
    texts_idx = [tokenizer.encode(text,add_special_tokens=False) for text in texts]
    for text_idx in texts_idx:text_idx.append(args.emb_id)
    max_len = max([len(text_idx) for text_idx in texts_idx])
    texts_idx = [text_idx + [args.pad_id]*(max_len-len(text_idx)) for text_idx in texts_idx]
    print(texts_idx)
    
    #find the mask positions in texts_idx
    mask_positions = []
    for text_idx in texts_idx:
        mask_positions.append([i for i, x in enumerate(text_idx) if x == args.mask_id])
    print(mask_positions)
    input_ids = torch.tensor(texts_idx,dtype=torch.long,device=device)
    MAX_CUM_PROB = 0.7
    import time
    with torch.no_grad():
        with torch.autocast(device_type=device,dtype=torch.float32):
            print('start to forward[CPU]')
            start_time = time.time()
            logits = model.forward(input_ids)
            end_time = time.time()
            print(f'forward time is {end_time-start_time}')
            for b in range(len(texts_idx)):
                mask_position = mask_positions[b]
                masked_prob = torch.softmax(logits[b,mask_position],dim=-1)
                print(masked_prob)
                print(masked_prob.shape)
                probs,indices = torch.topk(masked_prob,10)
                print(probs)
                print(indices)
                for position in mask_position:
                    cum_prob = 0
                    mask_idx = 0
                    for i in range(10):
                        texts_idx[b][position] = indices[mask_idx][i].item()
                        prob = probs[mask_idx][i].item()
                        cum_prob += prob
                        print(tokenizer.decode(texts_idx[b]),' prob is ',prob,' cum_prob is ',cum_prob)
                        if cum_prob > MAX_CUM_PROB:
                            break
                    mask_idx += 1
                print('----------------------------------')
                # for i in range(10):
                #     mask_idx = 0
                #     cum_prob = 0
                #     for position in mask_position:
                #         texts_idx[b][position] = indices[mask_idx][i].item()
                #         cum_prob += probs[mask_idx][i].item()
                #         print('**********************************')
                #         print(tokenizer.decode(texts_idx[b]),' prob is ',probs[mask_idx][i].item(),' cum_prob is ',cum_prob)
                #         print('**********************************')
                #         mask_idx += 1
                #         if cum_prob > MAX_CUM_PROB:
                #             break
                # print('----------------------------------')

