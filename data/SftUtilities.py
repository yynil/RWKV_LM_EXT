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

if __name__ == '__main__r':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_dir', type=str, default='/media/yueyulin/data_4t/data/tigerbot_sft_dataset/')
    args = parser.parse_args()
    import os
    ds = []
    for file in os.listdir(args.ds_dir):
        if file.endswith('_dataset'):
            dataset = datasets.load_from_disk(os.path.join(args.ds_dir, file))
            print(dataset)
            ds.append(dataset)
    ds = datasets.concatenate_datasets(ds)
    print('-------------------')
    print(ds)
    import torch
    def data_collator(batch):
        input_ids = [b['input_ids'] for b in batch]
        labels = [b['labels'] for b in batch]
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        return input_ids, labels
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(ds,shuffle=False, pin_memory=True, batch_size=1, num_workers=1, persistent_workers=False, drop_last=True, collate_fn=data_collator)
    for i, batch in enumerate(train_dataloader):
        print(i)
        input_ids, labels = batch
        print(input_ids)
        print(labels)
        print(input_ids.shape)
        print(labels.shape)
        print('-------------------')
        if i == 4:
            break
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    print(tokenizer.decode([ 3319,   145,   178,    59, 14389, 16771, 16728, 19151, 11405, 12732,
         10675, 17303, 17389, 14315, 23244, 12642, 10250, 10696, 16136, 11011,
         11459, 11043, 19134,   261,  3319,   165,   151,    59, 16771, 12221,
         19151, 10894, 11647]))
    print(tokenizer.decode([ 16771, 12221, 19151,
         10894, 11647]))
if __name__ == '__main__create_batch':
    parent_dir = '/media/yueyulin/data_4t/data/tigerbot_sft_dataset/'
    sizes = [64,128,256,512,1024,2048]
    from custom_datasets import read_dataset
    ds = read_dataset(parent_dir, sizes)
    print(ds[0])
if __name__ == '__main__2':
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
        if file.endswith('.json'):
            input_jsonl = os.path.join(args.input_dir, file)
            output_sft = os.path.join(args.output_dir, file.replace('.json', '_dataset_'))
            create_variable_sized_sft_from_jsonl(input_jsonl, output_sft, args.tokenizer_file)

if __name__ == '__main__':
    input_jsonl = '/home/yueyulin/下载/ft-data-conversion_output.jsonl'
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    create_variable_sized_sft_from_jsonl(input_jsonl, '/home/yueyulin/下载/ft-data-conversion-ds_',tokenizer_file)

if __name__ == '__main_test__':
    input_jsonl = '/media/yueyulin/data_4t/data/tigerbot_sft/tigerbot-alpaca-zh-0.5m.json'
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    create_sft_from_jsonl(input_jsonl, 'data/summarization.sft',tokenizer_file)
    # examples = {'input_ids': [3319, 145, 178, 59, 17109, 10673, 10250, 10283, 14600, 12325, 14734, 11043, 15044, 11124, 14035, 11098, 16417, 19137, 14583, 12604, 16721, 14600, 12325, 14734, 15190, 16503, 10821, 12484, 13365, 13445, 28329, 14600, 12325, 11043, 15044, 19151, 17777, 10746, 17136, 11, 14035, 11098, 19151, 16102, 11723, 2453, 13717, 17116, 18005, 10079, 10450, 10414, 2453, 15325, 11979, 17136, 11766, 10079, 15260, 13570, 11979, 2453, 15577, 12363, 17384, 13327, 10690, 10079, 11039, 11672, 11641, 2453, 14746, 11979, 13033, 10079, 14569, 11039, 2453, 18157, 11164, 13227, 11, 3319, 165, 151, 59, 10086, 17777, 10746, 17136, 10087, 16679, 17162, 10333, 10250, 10283, 11043, 11017, 13240, 10660, 2453, 16102, 10788, 14734, 17170, 10415, 10709, 10370, 10985, 10988, 10842, 11043, 10292, 9822, 17777, 10746, 17136, 16664, 10775, 9823, 14734, 10409, 10843, 19137, 11454, 10409, 10843, 10285, 10390, 11967, 16503, 13234, 10792, 14057, 11632, 12705, 13085, 14484, 10257, 17149, 16403, 10250, 17999, 10409, 10843, 10080, 11454, 11640, 15090, 15887, 10336, 12985, 10285, 19137, 13240, 10660, 14734, 10994, 15809, 11645, 10985, 10333, 16403, 10846, 15752, 10838, 19137, 10444, 11454, 14057, 11632, 12705, 13085, 14484, 10257, 19137, 10390, 10460, 10292, 10250, 11043, 9822, 17777, 10746, 17136, 9823, 12604, 10841, 11960, 15878, 19137, 12202, 10261, 12307, 11459, 17303, 16109, 10355, 10257, 10333, 13192, 10993, 10080, 12307, 11459, 14734, 10250, 15033, 11920, 13046, 11017, 10588, 9822, 15462, 15597, 9823, 13046, 19137, 13091, 10250, 15604, 17038, 18216, 16947, 17141, 10256, 15260, 14734, 16155, 14746, 15708, 10370, 12314, 14583, 14328, 19137, 11124, 10687, 10390, 10846, 13407, 14328, 17311, 15287, 14987, 15458, 10730, 11124, 12360, 12415, 18216, 11984, 10080, 13240, 10660, 11124, 12307, 11459, 14734, 15462, 15597, 13046, 10370, 12266, 11726, 17149, 16403, 11038, 10460, 19137, 10995, 12676, 10250, 11920, 11043, 10292, 9822, 16909, 13981, 10997, 12816, 10678, 11029, 9823, 14734, 11638, 10412, 10270, 19137, 10412, 10270, 16707, 11436, 16056, 11001, 14057, 11632, 12705, 13085, 14484, 10257, 10250, 15033, 14873, 14328, 16884, 9822, 10260, 11021, 12410, 16676, 14734, 10286, 11932, 9823, 10080, 14600, 12325, 15489, 13229, 13064, 19137, 14057, 11632, 12705, 13085, 14484, 16056, 12348, 10333, 15735, 10788, 19137, 13240, 10660, 12604, 10292, 10333, 15462, 15597, 13046, 14734, 10250, 11098, 19137, 10390, 10322, 17189, 17141, 10482, 14589, 15462, 15597, 13046, 14734, 10427, 15500, 13036, 12276, 46, 17152, 12824, 10687, 17688, 11647, 10997, 10261, 14583, 14328, 11038, 12604, 10452, 13234, 16362, 10673, 10333, 17148, 10283, 11920, 13046, 10080, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 10086, 17777, 10746, 17136, 10087, 16679, 17162, 10333, 10250, 10283, 11043, 11017, 13240, 10660, 2453, 16102, 10788, 14734, 17170, 10415, 10709, 10370, 10985, 10988, 10842, 11043, 10292, 9822, 17777, 10746, 17136, 16664, 10775, 9823, 14734, 10409, 10843, 19137, 11454, 10409, 10843, 10285, 10390, 11967, 16503, 13234, 10792, 14057, 11632, 12705, 13085, 14484, 10257, 17149, 16403, 10250, 17999, 10409, 10843, 10080, 11454, 11640, 15090, 15887, 10336, 12985, 10285, 19137, 13240, 10660, 14734, 10994, 15809, 11645, 10985, 10333, 16403, 10846, 15752, 10838, 19137, 10444, 11454, 14057, 11632, 12705, 13085, 14484, 10257, 19137, 10390, 10460, 10292, 10250, 11043, 9822, 17777, 10746, 17136, 9823, 12604, 10841, 11960, 15878, 19137, 12202, 10261, 12307, 11459, 17303, 16109, 10355, 10257, 10333, 13192, 10993, 10080, 12307, 11459, 14734, 10250, 15033, 11920, 13046, 11017, 10588, 9822, 15462, 15597, 9823, 13046, 19137, 13091, 10250, 15604, 17038, 18216, 16947, 17141, 10256, 15260, 14734, 16155, 14746, 15708, 10370, 12314, 14583, 14328, 19137, 11124, 10687, 10390, 10846, 13407, 14328, 17311, 15287, 14987, 15458, 10730, 11124, 12360, 12415, 18216, 11984, 10080, 13240, 10660, 11124, 12307, 11459, 14734, 15462, 15597, 13046, 10370, 12266, 11726, 17149, 16403, 11038, 10460, 19137, 10995, 12676, 10250, 11920, 11043, 10292, 9822, 16909, 13981, 10997, 12816, 10678, 11029, 9823, 14734, 11638, 10412, 10270, 19137, 10412, 10270, 16707, 11436, 16056, 11001, 14057, 11632, 12705, 13085, 14484, 10257, 10250, 15033, 14873, 14328, 16884, 9822, 10260, 11021, 12410, 16676, 14734, 10286, 11932, 9823, 10080, 14600, 12325, 15489, 13229, 13064, 19137, 14057, 11632, 12705, 13085, 14484, 16056, 12348, 10333, 15735, 10788, 19137, 13240, 10660, 12604, 10292, 10333, 15462, 15597, 13046, 14734, 10250, 11098, 19137, 10390, 10322, 17189, 17141, 10482, 14589, 15462, 15597, 13046, 14734, 10427, 15500, 13036, 12276, 46, 17152, 12824, 10687, 17688, 11647, 10997, 10261, 14583, 14328, 11038, 12604, 10452, 13234, 16362, 10673, 10333, 17148, 10283, 11920, 13046, 10080, 1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]}
    # print(len(examples['input_ids']),len(examples['labels']))
    # real_len_input_ids = len([i for i in examples['input_ids'] if i != 0])
    # first_index_of_label = 0
    # for i, label in enumerate(examples['labels']):
    #     if label != -100:
    #         first_index_of_label = i
    #         break
    # eos_index = examples['labels'].index(1)
    # print(real_len_input_ids,first_index_of_label,eos_index)
    # import sys
    # import os
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    # tokenizer = TRIE_TOKENIZER(tokenizer_file)
    # print(tokenizer.decode(examples['input_ids'][:real_len_input_ids]))
    # print('-------------------')
    # print(tokenizer.decode(examples['labels'][first_index_of_label:eos_index]))
