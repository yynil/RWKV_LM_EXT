import os
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(parent_path)
print(f'add path: {parent_path} to sys.path')
from infer.states_generator import StatesGenerator


if __name__ == '__main__':
    model_file = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'

    kg_states_file = '/media/yueyulin/data_4t/models/states_tuning/instructKGC_scattered/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    type_states_file = '/media/yueyulin/data_4t/models/states_tuning/kg_type/20240702-105004/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    sg = StatesGenerator(model_file,tokenizer_file)
    sg.load_states(kg_states_file,'kg')
    sg.load_states(type_states_file,'type')

    test_txt_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'test.txt')
    with open(test_txt_file,'r') as f:
        lines = f.readlines()
        txt = ''.join(lines)
    print(txt)
    from harvesttext import HarvestText as ht
    h = ht()
    sentences = h.cut_sentences(txt)
    typed_sentences = []
    current_type = None
    current_sentence = None
    import json
    from kg_schema import whole_schema,all_types
    print(all_types)
    kg_instruction = '你是一个图谱实体知识结构化专家。请从input中抽取出符合schema定义的实体实例和其属性，不存在的属性不输出，属性存在多值就返回列表。请按照JSON字符串的格式回答。'
    type_instruction = '请根据input中的文本内容，选择schema中的最符合input内容的类别输出，用JSON格式输出。'
    for s in sentences:
        print(s)
        input_text = {'input':s,'schema':all_types}
        input_text = json.dumps(input_text,ensure_ascii=False)
        type_output = sg.generate(input_text,type_instruction,'type',top_k=0,top_p=0,gen_count=128)
        print(type_output)
        try:
            result = json.loads(type_output)['result']
            print(f'infer {s} as {result}')
            if isinstance(result,list):
                result = result[0]
            if result != current_type:
                if current_type is not None:
                    typed_sentences.append((current_type,current_sentence))
                current_type = result
                current_sentence = s
            else:
                current_sentence += s
        except:
            print('error')
    if current_type is not None:
        typed_sentences.append((current_type,current_sentence))
    print(typed_sentences)

    for type,sentence in typed_sentences:
        type = '事件'
        print(f'processing {type} {sentence}')
        schema = whole_schema[type]
        input_text = json.dumps({'input':sentence,'schema':schema},ensure_ascii=False)
        kg_output = sg.generate(input_text,kg_instruction,'kg',top_k=0,top_p=0,gen_count=2048)
        print(kg_output)
        kg_output = json.loads(kg_output)
        print(kg_output['result'])
        print('---------------------------------')


