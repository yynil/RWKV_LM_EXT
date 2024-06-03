import datasets
import functools
cat_char = '🐱'
bot_char = '🤖'

def convert_2_conversation(examples, input_field = 'input',  instruction_field = 'instruction', output_field = 'output'):
    inputs = examples[input_field]
    instructions = examples[instruction_field]
    outputs = examples[output_field]
    input_texts = []
    target_texts = []
    for i, (input_text, instruction, output) in enumerate(zip(inputs, instructions, outputs)):
        input_text = input_text.strip()
        instruction = instruction.strip()
        output = output.strip()
        input_texts.append(f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:')
        target_texts.append(f'{output}')
    return {'input': input_texts, 'output': target_texts}

def tokenize_fn(examples, tokenizer,pad_id=0,eos_id=1,max_length=512,input_field = 'input', output_field = 'output'):
    inputs = examples[input_field]
    outputs = examples[output_field]
    inputs_ids = []
    targets_ids = []
    for i, (input_text, output) in enumerate(zip(inputs, outputs)):
        input_ids = tokenizer.encode(input_text)
        output_ids = tokenizer.encode(output)
        whole_ids = input_ids + output_ids
        shifted_output_ids = [-100]*(len(input_ids)-1) + output_ids+[eos_id]
        length = len(whole_ids)
        if length > max_length:
            whole_ids = whole_ids[:max_length]
            shifted_output_ids = shifted_output_ids[:max_length]
        else:
            whole_ids += [pad_id]*(max_length-length)
            shifted_output_ids += [-100]*(max_length-length)
        inputs_ids.append(whole_ids)
        targets_ids.append(shifted_output_ids)
    return {'input_ids': inputs_ids, 'labels': targets_ids}

def tokenize_fn_no_chunk(examples, tokenizer,pad_id=0,eos_id=1,input_field = 'input', output_field = 'output'):
    inputs = examples[input_field]
    outputs = examples[output_field]
    inputs_ids = []
    targets_ids = []
    for i, (input_text, output) in enumerate(zip(inputs, outputs)):
        input_ids = tokenizer.encode(input_text)
        output_ids = tokenizer.encode(output)
        whole_ids = input_ids + output_ids
        shifted_output_ids = [-100]*(len(input_ids)-1) + output_ids+[eos_id]
        inputs_ids.append(whole_ids)
        targets_ids.append(shifted_output_ids)
    return {'input_ids': inputs_ids, 'labels': targets_ids}

def create_variable_sized_sft_from_jsonl(input_jsonl, output_sft,tokenizer_file,input_field = 'input',  instruction_field = 'instruction', output_field = 'output'):
    ds = datasets.load_dataset('json', data_files=input_jsonl)['train']
    map_fn = functools.partial(convert_2_conversation, input_field=input_field, instruction_field=instruction_field, output_field=output_field)
    ds = ds.map(map_fn, batched=True, num_proc=4, remove_columns=ds.column_names)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    map_fn = functools.partial(tokenize_fn_no_chunk, tokenizer=tokenizer, input_field='input', output_field='output')
    ds = ds.map(map_fn, batched=True, num_proc=1,remove_columns=ds.column_names)
    sizes = [64,128,256,512,1024,2048]
    data = [[] for i in range(len(sizes)+1)]
    import bisect
    for i, example in enumerate(ds):
        length = len(example['input_ids'])
        idx = bisect.bisect_left(sizes, length)
        if idx < len(sizes):
            data[idx].append(example)
    
    for i in range(len(data)):
        ds_name = output_sft + f'ds_{sizes[i] if i < len(sizes) else "max"}'
        if len(data[i]) > 0:
            data_dict = {'input_ids': [], 'labels': []}
            for d in data[i]:
                data_dict['input_ids'].append(d['input_ids'])
                data_dict['labels'].append(d['labels'])
            dataset = datasets.Dataset.from_dict(data_dict)
            dataset.save_to_disk(ds_name)
            print(f'save {ds_name} with {len(dataset)} examples')

def create_sft_from_jsonl(input_jsonl, output_sft,tokenizer_file,input_field = 'input',  instruction_field = 'instruction', output_field = 'output'):
    ds = datasets.load_dataset('json', data_files=input_jsonl)['train']
    map_fn = functools.partial(convert_2_conversation, input_field=input_field, instruction_field=instruction_field, output_field=output_field)
    ds = ds.map(map_fn, batched=True, num_proc=4, remove_columns=ds.column_names)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    map_fn = functools.partial(tokenize_fn, tokenizer=tokenizer, input_field='input', output_field='output')
    ds = ds.map(map_fn, batched=True, num_proc=1,remove_columns=ds.column_names)
    ds.save_to_disk(output_sft)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/media/yueyulin/data_4t/data/tigerbot_sft/')
    parser.add_argument('--output_dir', type=str, default='/media/yueyulin/data_4t/data/tigerbot_sft_dataset/')
    parser.add_argument('--tokenizer_file', type=str, default='/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt')
    args = parser.parse_args()
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    import sys
    tokenizer_file_parent_dir = os.path.dirname(os.path.dirname(args.tokenizer_file))
    sys.path.append(tokenizer_file_parent_dir)

    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(args.tokenizer_file)
    print(tokenizer)
    for file in os.listdir(args.input_dir):
        if file.endswith('.json') or file.endswith('.jsonl'):
            input_jsonl = os.path.join(args.input_dir, file)
            output_sft = os.path.join(args.output_dir, file.replace('.json', '_dataset_'))
            create_variable_sized_sft_from_jsonl(input_jsonl, output_sft, args.tokenizer_file)

