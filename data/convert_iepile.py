import argparse
import os
import orjson

CACHE_SIZE = 1024
def convert_ie_2_instruction(input_jsonl :str, output_fp, cache_list,task):
    if input_jsonl == 'FLUSH':
        print('flush')
        for item in cache_list:
            output_fp.write(orjson.dumps(item).decode() + '\n')
        cache_list.clear()
        output_fp.close()
        print(f'close {output_fp.name}')
        return
    data = orjson.loads(input_jsonl)
    if 'instruction' in data and 'output' in data:
        data_task = data['task']
        if data_task != task:
            return
        instruction = data['instruction']
        output = data['output']
        instruction_data = orjson.loads(instruction)
        instruction = instruction_data['instruction']
        input_text = instruction_data['input']
        schema = instruction_data['schema']
        input_text = orjson.dumps({'input': input_text, 'schema': schema}).decode()
    elif 'input' in data and 'output' in data:
        input_text = data['input']
        output = data['output']
        instruction = '你是专门进行关系抽取的专家。请从input中抽取关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。'
    else:
        raise ValueError(f'input jsonl should contain either instruction and output or input and output {input_jsonl}')
    cache_list.append({'input': input_text, 'instruction': instruction, 'output': output})
    if len(cache_list) >= CACHE_SIZE:
        for item in cache_list:
            output_fp.write(orjson.dumps(item).decode() + '\n')
        cache_list.clear()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--task', type=str, default='NER')
    args = parser.parse_args()
    output_fp = open(args.output_jsonl, 'w')
    with open(args.input_jsonl, 'r') as fp:
        cache_list = []
        for line in fp:
            convert_ie_2_instruction(line, output_fp, cache_list,args.task)
    convert_ie_2_instruction('FLUSH', output_fp, cache_list,args.task)
    print(f'finish {args.output_jsonl}')
