from copy import deepcopy
import random
import torch
tokenizer = None
def whole_word_mask(token_ids, mlm_probability,segments = None):
    if segments is None:
        mask = []
        for token in token_ids:
            if random.random() < mlm_probability:
                mask.append(1)
            else:
                mask.append(0)
        return mask
    else:
        mask = []
        offset = 0
        for i in range(len(segments)):
            if random.random() < mlm_probability:
                mask.extend([1]*len(segments[i]))
            else:
                mask.extend([0]*len(segments[i]))
        return mask

def dup_mae_collator(examples,
                     max_seq_length,
                     encoder_mlm_probability,
                     mask_id = 3,
                     emb_id = 1,
                     vocab_size = 65536,
                     pad_id = 0):
    batch = {
        'encoder_input_ids': [],
        'encoder_labels': [],
        'decoder_input_ids': [],
        'decoder_labels': [],
        'bag_word_weight': []
    }
    tgt_len = max_seq_length - 1
    for example in examples:
        token_ids = example['token_ids']
        segment_ids = example['segment_ids'] if 'segment_ids' in example else None
        encoder_input_ids = token_ids[:tgt_len]
        encoder_input_ids.append(emb_id)
            
        padding_size = max_seq_length - len(encoder_input_ids)
        mask = whole_word_mask(encoder_input_ids[:-1], encoder_mlm_probability,segments = segment_ids)

        encoder_labels = deepcopy(encoder_input_ids)
        decoder_labels = deepcopy(encoder_input_ids)
        decoder_input_ids = deepcopy(encoder_input_ids)
        for i, m in enumerate(mask):
            if i >= tgt_len:
                break#in case for some extem cases
            if m == 1:
                encoder_input_ids[i] = mask_id
            else:
                encoder_labels[i] = -100

        weight = torch.zeros(size=(vocab_size,))
        for t in example['token_ids'][:tgt_len]:
            weight[t] = 1 / len(example['token_ids'][:tgt_len])
        batch['bag_word_weight'].append(weight.unsqueeze(0))
        encoder_labels[-1] = -100
        decoder_labels[-1] = -100
        batch['encoder_input_ids'].append(encoder_input_ids if padding_size == 0 else encoder_input_ids + [pad_id]*padding_size)
        batch['encoder_labels'].append(encoder_labels if padding_size == 0 else encoder_labels + [-100]*padding_size)
        batch['decoder_input_ids'].append(decoder_input_ids if padding_size == 0 else decoder_input_ids + [pad_id]*padding_size)
        batch['decoder_labels'].append(decoder_labels if padding_size == 0 else decoder_labels + [-100]*padding_size)
    batch['encoder_input_ids'] = torch.tensor(batch['encoder_input_ids'],dtype=torch.long)
    batch['encoder_labels'] = torch.tensor(batch['encoder_labels'],dtype=torch.long)
    batch['decoder_input_ids'] = torch.tensor(batch['decoder_input_ids'],dtype=torch.long)
    batch['decoder_labels'] = torch.tensor(batch['decoder_labels'],dtype=torch.long)
    batch['bag_word_weight'] = torch.cat(batch['bag_word_weight'],dim=0)

    return batch

def mae_collator(examples, 
                 max_seq_length, 
                 encoder_mlm_probability, 
                 mask_id = 3,
                 emb_id = 1,
                 pad_id = 0):
    batch = {
        'encoder_input_ids': [],
        'encoder_labels': [],
        'decoder_input_ids': [],
        'decoder_labels': []
    }
    tgt_len = max_seq_length - 1
    for example in examples:
        token_ids = example['token_ids']
        segment_ids = example['segment_ids'] if 'segment_ids' in example else None
        encoder_input_ids = token_ids[:tgt_len]
        encoder_input_ids.append(emb_id)
            
        padding_size = max_seq_length - len(encoder_input_ids)
        mask = whole_word_mask(encoder_input_ids[:-1], encoder_mlm_probability,segments = segment_ids)

        encoder_labels = deepcopy(encoder_input_ids)
        decoder_labels = deepcopy(encoder_input_ids)
        decoder_input_ids = deepcopy(encoder_input_ids)
        for i, m in enumerate(mask):
            if m == 1:
                encoder_input_ids[i] = mask_id
            else:
                encoder_labels[i] = -100
        encoder_labels[-1] = -100
        decoder_labels[-1] = -100
        batch['encoder_input_ids'].append(encoder_input_ids if padding_size == 0 else encoder_input_ids + [pad_id]*padding_size)
        batch['encoder_labels'].append(encoder_labels if padding_size == 0 else encoder_labels + [-100]*padding_size)
        batch['decoder_input_ids'].append(decoder_input_ids if padding_size == 0 else decoder_input_ids + [pad_id]*padding_size)
        batch['decoder_labels'].append(decoder_labels if padding_size == 0 else decoder_labels + [-100]*padding_size)
    batch['encoder_input_ids'] = torch.tensor(batch['encoder_input_ids'],dtype=torch.long)
    batch['encoder_labels'] = torch.tensor(batch['encoder_labels'],dtype=torch.long)
    batch['decoder_input_ids'] = torch.tensor(batch['decoder_input_ids'],dtype=torch.long)
    batch['decoder_labels'] = torch.tensor(batch['decoder_labels'],dtype=torch.long)

    return batch

def mlm_collator_with_segment_ids(examples, 
                 max_seq_length, 
                 encoder_mlm_probability, 
                 mask_id = 3,
                 emb_id = 1,):
    batch = {
        'encoder_input_ids': [],
        'encoder_labels': []
    }
    tgt_len = max_seq_length - 1
    for example in examples:
        token_ids = example['token_ids']
        segment_ids = example['segment_ids'] if 'segment_ids' in example else None
        encoder_input_ids = token_ids[:tgt_len]
        encoder_input_ids.append(emb_id)
            
        padding_size = max_seq_length - len(encoder_input_ids)
        mask = whole_word_mask(encoder_input_ids[:-1], encoder_mlm_probability,segments = segment_ids)

        encoder_labels = deepcopy(encoder_input_ids)
        for i, m in enumerate(mask):
            if m == 1:
                encoder_input_ids[i] = mask_id
            else:
                encoder_labels[i] = -100
        encoder_labels[-1] = -100
        batch['encoder_input_ids'].append(encoder_input_ids if padding_size == 0 else encoder_input_ids + [0]*padding_size)
        batch['encoder_labels'].append(encoder_labels if padding_size == 0 else encoder_labels + [-100]*padding_size)
    batch['encoder_input_ids'] = torch.tensor(batch['encoder_input_ids'],dtype=torch.long)
    batch['encoder_labels'] = torch.tensor(batch['encoder_labels'],dtype=torch.long)

    return batch

def mlm_collator(examples, 
                 max_seq_length, 
                 encoder_mlm_probability, 
                 mask_id = 3,
                 emb_id = 1,
                 pad_id = 0):
    batch = {
        'encoder_input_ids': [],
        'encoder_labels': []
    }
    tgt_len = max_seq_length - 1
    for example in examples:
        token_ids = example['token_ids']
        encoder_input_ids = token_ids[:tgt_len]
        encoder_input_ids.append(emb_id)
            
        padding_size = max_seq_length - len(encoder_input_ids)
        mask = whole_word_mask(encoder_input_ids[:-1], encoder_mlm_probability,segments = None)

        encoder_labels = deepcopy(encoder_input_ids)
        for i, m in enumerate(mask):
            if m == 1:
                encoder_input_ids[i] = mask_id
            else:
                encoder_labels[i] = -100
        encoder_labels[-1] = -100
        batch['encoder_input_ids'].append(encoder_input_ids if padding_size == 0 else encoder_input_ids + [pad_id]*padding_size)
        batch['encoder_labels'].append(encoder_labels if padding_size == 0 else encoder_labels + [-100]*padding_size)
    batch['encoder_input_ids'] = torch.tensor(batch['encoder_input_ids'],dtype=torch.long)
    batch['encoder_labels'] = torch.tensor(batch['encoder_labels'],dtype=torch.long)

    return batch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='input_file',default='/media/yueyulin/data_4t/data/ccia_ds_mini_128/')
    parser.add_argument('--max_length',type=int,default=128)
    parser.add_argument('--vocab_size',type=int,default=151343)
    args = parser.parse_args()
    import datasets
    from datasets import load_from_disk
    dataset = load_from_disk(args.data_dir)
    print(dataset)
    print(dataset[0])
    from torch.utils.data import DataLoader
    from functools import partial
    collator = partial(mae_collator, max_seq_length=args.max_length, encoder_mlm_probability=0.3,pad_id=151334,mask_id=151330,emb_id=151329)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in dataloader:
        encoder_input_ids,encoder_labels,decoder_input_ids,decoder_labels = batch['encoder_input_ids'],batch['encoder_labels'],batch['decoder_input_ids'],batch['decoder_labels']
        print(encoder_input_ids)
        print(encoder_labels)
        print(decoder_input_ids)
        print(decoder_labels)
        print(encoder_input_ids.shape)
        print(encoder_labels.shape)
        print(decoder_input_ids.shape)
        print(decoder_labels.shape)
        break

    collator = partial(dup_mae_collator, max_seq_length=args.max_length, encoder_mlm_probability=0.3,vocab_size=args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in dataloader:
        encoder_input_ids,encoder_labels,decoder_input_ids,decoder_labels,bag_word_weight = batch['encoder_input_ids'],batch['encoder_labels'],batch['decoder_input_ids'],batch['decoder_labels'],batch['bag_word_weight']
        print(encoder_input_ids)
        print(encoder_labels)
        print(decoder_input_ids)
        print(decoder_labels)
        print(bag_word_weight)
        print(encoder_input_ids.shape)
        print(encoder_labels.shape)
        print(decoder_input_ids.shape)
        print(decoder_labels.shape)
        print(bag_word_weight.shape)
        break

    collator = partial(mlm_collator, max_seq_length=args.max_length, encoder_mlm_probability=0.15)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in dataloader:
        encoder_input_ids,encoder_labels = batch['encoder_input_ids'],batch['encoder_labels']
        print(encoder_input_ids.tolist())
        print(encoder_labels.tolist())
        print(encoder_input_ids.shape)
        print(encoder_labels.shape)
        break
