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
    os.environ['WKV'] = ''
    os.environ['RWKV_TRAIN_TYPE'] = ''
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    os.environ['NO_CUDA'] = '1'
setup_env()
from transformers import AutoTokenizer
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
def test_texts(args, model, device, texts, tokenizer):
    texts_idx = [tokenizer.encode(text,add_special_tokens=False) for text in texts]
    for text_idx in texts_idx:text_idx.append(args.emb_id)
    max_len = max([len(text_idx) for text_idx in texts_idx])
    texts_idx = [text_idx + [args.pad_id]*(max_len-len(text_idx)) for text_idx in texts_idx]
    
   
    input_ids = torch.tensor(texts_idx,dtype=torch.long,device=device)
    MAX_CUM_PROB = 0.7
    import time
    from sentence_transformers.util import cos_sim as sim_fn
    with torch.no_grad():
        with torch.autocast(device_type=device,dtype=torch.float32):
            print('start to forward[CPU]')
            start_time = time.time()
            embs = model.encode_sentence(input_ids)
            end_time = time.time()
            print(f'forward time is {end_time-start_time}')
            print(sim_fn(embs[0],embs[1:]).squeeze(0).tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test MLM model")
    parser.add_argument("--model_file",type=str,default='/media/yueyulin/KINGSTON/models/macro_zh_bi_encoder/trainable_model/epoch_21/RWKV-x060-MLM-ctx4096.pth.pth')
    args = parser.parse_args() 
    print(args)
    model = load_base_model(args.model_file)

    args.emb_id = 151329
    args.pad_id = 151334
    args.mask_id = 151330
    device = 'cpu'
    model = model.to(device=device,dtype=torch.float32)
    texts = ['每天吃苹果有什么好处？',
             '某些水果和蔬菜特别富含可溶性纤维。在水果中，柑橘类水果如橙子、葡萄柚和柠檬的可溶性纤维含量相当高，浆果也是如此，包括草莓、蓝莓、黑莓和覆盆子。苹果和梨也提供可溶性纤维，香蕉也是如此。',
             '苹果，落叶乔木，叶子椭圆形，花白色带有红晕。果实圆形，味甜或略酸，是常见水果，具有丰富营养成分，有食疗、辅助治疗功能。苹果原产于欧洲、中亚、西亚和土耳其一带，于十九世纪传入中国。中国是世界最大的苹果生产国，在东北、华北、华东、西北和四川、云南等地均有栽培。',
             '苹果生吃治便秘，熟吃治腹泻：苹果中含有丰富的鞣酸、果胶、膳食纤维等特殊物质，鞣酸是肠道收敛剂，它能减少肠道分泌而使大便内水分减少，从而止泻。而果胶则是个“两面派”，未经加热的生果胶有软化大便缓解便秘的作用，煮过的果胶却摇身一变，具有收敛、止泻的功效。膳食纤维又起到通便作用。']
    tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
    test_texts(args, model, device, texts, tokenizer)
    

    texts = ['每天吃苹果有什么好处？',
             '宁神安眠：苹果中含有的磷和铁等元素，易被肠壁吸收，有补脑养血、宁神安眠作用。苹果的香气是治疗抑郁和压抑感的良药。研究发现，在诸多气味中，苹果的香气对人的心理影响最大，它具有明显的消除心理压抑感的作用。',
             '美白养颜、降低胆固醇：苹果中的胶质和微量元素铬能保持血糖的稳定，还能有效地降低胆固醇。苹果中的粗纤维可促进肠胃蠕功，并富含铁、锌等微量元素，可使皮肤细润有光泽，起到美容瘦身的作用。',
             '苹果生吃治便秘，熟吃治腹泻：苹果中含有丰富的鞣酸、果胶、膳食纤维等特殊物质，鞣酸是肠道收敛剂，它能减少肠道分泌而使大便内水分减少，从而止泻。而果胶则是个“两面派”，未经加热的生果胶有软化大便缓解便秘的作用，煮过的果胶却摇身一变，具有收敛、止泻的功效。膳食纤维又起到通便作用。',
             '保护心脏：苹果的纤维、果胶、抗氧化物等能降低体内坏胆固醇并提高好胆固醇含量，所以每天吃一两个苹果不容易得心脏病。']
    test_texts(args, model, device, texts, tokenizer)
    
    texts = ['庆余年2是谁投资拍摄的？',
             '《庆余年第二季》是由孙皓执导，王倦担任编剧，张若昀、李沁领衔主演，陈道明特别主演，吴刚、田雨领衔主演，袁泉、毛晓彤特邀出演，郭麒麟特邀主演的古装传奇剧 [1] [65]。',
             '该剧于2024年5月16日在央视八套首播，腾讯视频全网独播 [60]。2024年5月16日，据“CCTV电视剧”官微，数据显示，CCTV-8黄金强档热播剧《庆余年2》当晚实时直播关注度峰值破2 [57]；5月28日，据灯塔专业版数据，《庆余年2》累计正片播放量已突破12亿 [72]。',
             '苹果生吃治便秘，熟吃治腹泻：苹果中含有丰富的鞣酸、果胶、膳食纤维等特殊物质，鞣酸是肠道收敛剂，它能减少肠道分泌而使大便内水分减少，从而止泻。而果胶则是个“两面派”，未经加热的生果胶有软化大便缓解便秘的作用，煮过的果胶却摇身一变，具有收敛、止泻的功效。膳食纤维又起到通便作用。',
             '《庆余年第二季》出品公司：中央电视台、上海腾讯企鹅影视文化传播有限公司、天津阅文影视文化传媒有限公司、新丽电视文化投资有限公司、新丽（上海）影视有限公司']
    test_texts(args, model, device, texts, tokenizer)