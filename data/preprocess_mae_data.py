import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(f'appended {parent_dir} to sys.path')
import glob
import os
#os.environ['HF_DATASETS_CACHE'] = '/media/yueyulin/data_4t/cache'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import datasets
import random
import functools
tokenizer = None
ht = None


def tokenize_chinese(text,tokenizer_file):
    global tokenizer
    if tokenizer is None:
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
        tokenizer = TRIE_TOKENIZER(tokenizer_file)
    global ht
    if ht is None:
        from harvesttext import HarvestText
        ht = HarvestText()
    segments = ht.seg(text)
    input_ids = []
    segment_ids = []
    for seg in segments:
        ids = tokenizer.encode(seg)
        segment_ids.append(ids)
        input_ids.extend(ids)
    return input_ids, segment_ids, segments

def create_cci2_dataset(cci2_dir,
                            tokenizer_file,
                            max_seq_length: int,
                            short_seq_prob: float = 0.0):
    parquet_files = glob.glob(os.path.join(cci2_dir, '*.parquet'))
    print(f'Found {len(parquet_files)} parquet files in {cci2_dir}')
    
    target_length = max_seq_length - 1
    def cci2_tokenize_function(examples):
        sentences_ids = []
        segments_ids = []
        for sents in examples['sentences']:
            for sent in sents:
                input_ids, segment_ids, _ = tokenize_chinese(sent, tokenizer_file)
                sentences_ids.append(input_ids)
                segments_ids.append(segment_ids)
        return {"input_ids": sentences_ids, "segment_ids": segments_ids}

    def sentence_cci2(examples):
        global ht
        if ht is None:
            from harvesttext import HarvestText
            ht = HarvestText()
        sentences = ht.cut_sentences(examples["content"])
        return {"sentences": sentences}

    def cci2_pad_each_line(examples):
        blocks = []
        all_segs = []
        for sent,segs in zip(examples['input_ids'],examples['segment_ids']):
            curr_block = []
            current_segs = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            if len(curr_block)+len(sent) >= curr_tgt_len:
                # curr_block.append(emb_id)
                blocks.append(curr_block)
                all_segs.append(current_segs)
                curr_block = []
                current_segs = []
                curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            curr_block.extend(sent)
            current_segs.extend(segs)
        if len(curr_block) > 0:
            # curr_block.append(emb_id)
            blocks.append(curr_block)
            all_segs.append(current_segs)
        return {'token_ids': blocks, 'segment_ids': all_segs}
    ds = datasets.load_dataset('parquet', data_files=parquet_files)['train']
    print(f'Loaded dataset with {len(ds)} samples')
    print('seg sentence')
    cci2 = ds.map(sentence_cci2, num_proc=8, remove_columns=["content","id"])
    print('tokenize and seg words')
    tokenized_cci2 = cci2.map(cci2_tokenize_function, num_proc=8, batched=True, remove_columns=["sentences"])
    print('group lines')
    processed_cci2 = tokenized_cci2.map(cci2_pad_each_line, num_proc=8, batched=True, remove_columns=tokenized_cci2.column_names)
    return processed_cci2

def create_wiki_zh_dataset(wiki_dir,
                         tokenizer_file,
                         max_seq_length: int,
                         short_seq_prob: float = 0.0):
    parquet_files = glob.glob(os.path.join(wiki_dir, '*.arrow'))
    print(f'Found {len(parquet_files)} arrow files in {wiki_dir}')
    ds = datasets.load_dataset('arrow', data_files=parquet_files)['train']
    print(f'Loaded dataset with {len(ds)} samples')


    target_length = max_seq_length - 1
    def wiki_tokenize_function(examples):
        sentences = []
        segments = []
        for sents in examples['sentences']:
            current_seg = []
            current_sents = []
            for sent in sents:
                input_ids, segment_ids, _ = tokenize_chinese(sent,tokenizer_file)
                current_sents.append(input_ids)
                current_seg.append(segment_ids)
            sentences.append(current_sents)
            segments.append(current_seg)
        return {"input_ids": sentences, "segment_ids": segments}

    def sentence_wiki(examples):
        global ht
        if ht is None:
            from harvesttext import HarvestText
            ht = HarvestText()
        sentences = ht.cut_sentences(examples["text"])
        return {"sentences": sentences}

    def wiki_pad_each_line(examples):
        blocks = []
        all_segements = []
        for sents,segments in zip(examples['input_ids'],examples['segment_ids']):
            curr_block = []
            current_seg = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            for sent,seg in zip(sents,segments):
                if len(curr_block)+len(sent) >= curr_tgt_len:
                    # curr_block.append(emb_id)
                    blocks.append(curr_block)
                    all_segements.append(current_seg)
                    curr_block = []
                    current_seg = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_block.extend(sent)
                current_seg.extend(seg)
            if len(curr_block) > 0:
                # curr_block.append(emb_id)
                blocks.append(curr_block)
                all_segements.append(current_seg)
        return {'token_ids': blocks, 'segment_ids': all_segements}
    print('seg sentence')
    wiki = ds.map(sentence_wiki, num_proc=16, remove_columns=["title", "text"])
    print('tokenize and seg words')
    tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=16, batched=True, remove_columns=["sentences"])
    print('group lines')
    processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=16, batched=True, remove_columns=tokenized_wiki.column_names)
    return processed_wiki

def create_wiki_dataset(wiki_dir,
                         tokenizer_file,
                         max_seq_length: int,
                         short_seq_prob: float = 0.0):
    parquet_files = glob.glob(os.path.join(wiki_dir, '*.parquet'))
    print(f'Found {len(parquet_files)} parquet files in {wiki_dir}')
    ds = datasets.load_dataset('parquet', data_files=parquet_files)['train']
    print(f'Loaded dataset with {len(ds)} samples')
    import nltk
    # nltk.download('punkt')

    target_length = max_seq_length - 1
    def wiki_tokenize_function(examples):
        global tokenizer
        if tokenizer is None:
            from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
            tokenizer = TRIE_TOKENIZER(tokenizer_file)
        sentences = []
        for sents in examples['sentences']:
            sentences.append([tokenizer.encode(sent) for sent in sents])
        return {"input_ids": sentences}

    def sentence_wiki(examples):
        sentences = nltk.sent_tokenize(examples["text"])
        return {"sentences": sentences}

    def wiki_pad_each_line(examples):
        blocks = []
        for sents in examples['input_ids']:
            curr_block = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            for sent in sents:
                if len(curr_block)+len(sent) >= curr_tgt_len:
                    # curr_block.append(emb_id)
                    blocks.append(curr_block)
                    curr_block = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_block.extend(sent)
            if len(curr_block) > 0:
                # curr_block.append(emb_id)
                blocks.append(curr_block)
        return {'token_ids': blocks}
    wiki = ds.map(sentence_wiki, num_proc=16, remove_columns=["title", "text"])
    tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=16, batched=True, remove_columns=["sentences"])
    processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=1, batched=True, remove_columns=tokenized_wiki.column_names)
    return processed_wiki
def creat_book_dataset(book_dir,
                       tokenizer_file,
                       max_seq_length: int,
                        short_seq_prob: float = 0.0):
    txt_files = glob.glob(os.path.join(book_dir, '*.txt'))
    target_length = max_seq_length - 1# we need to add a emb_id to the end of each block
    print(f'Found {len(txt_files)} txt files in {book_dir}')
    ds = datasets.load_dataset('text', data_files=txt_files)['train']
    print(f'Loaded dataset with {len(ds)} samples')
    print(ds[0])
    def book_tokenize_function(examples):
        global tokenizer
        if tokenizer is None:
            from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
            tokenizer = TRIE_TOKENIZER(tokenizer_file)
        input_ids = []
        for text in examples['text']:
            token_ids = tokenizer.encode(text)
            input_ids.append(token_ids)
        return {'input_ids': input_ids}

    def book_pad_each_line(examples):
        blocks = []
        curr_block = []

        curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
        for sent in examples['input_ids']:
            if len(curr_block) + len(sent) >= curr_tgt_len:
                # curr_block.append(emb_id)
                blocks.append(curr_block)
                curr_block = []
                curr_tgt_len = target_length if random.random() > short_seq_prob \
                    else random.randint(3, target_length)
            curr_block.extend(sent)
        if len(curr_block) > 0:
            # curr_block.append(emb_id)
            blocks.append(curr_block)
        return {'token_ids': blocks}
    tokenized_bookcorpus = ds.map(book_tokenize_function, num_proc=16, remove_columns=["text"], batched=True)
    processed_bookcorpus = tokenized_bookcorpus.map(book_pad_each_line, num_proc=16, batched=True,
                                                    remove_columns=["input_ids"])
    return processed_bookcorpus
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki_dir', type=str,default=None)
    parser.add_argument('--book_dir', type=str,default=None)
    parser.add_argument('--wiki_zh_dir', type=str,default=None)
    parser.add_argument('--output_dir', type=str,default='/media/yueyulin/data_4t/data/wikizh_mae_dataset')
    parser.add_argument('--cci2_dir',type=str,default=None)
    parser.add_argument('--tokenizer_file', type=str,default='/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt')
    args = parser.parse_args()
    
    ds = []
    if args.wiki_dir is not None:
        wiki_dataset = create_wiki_dataset(args.wiki_dir, args.tokenizer_file, 512)
        print(wiki_dataset)
        print(wiki_dataset[0])
        print('-----------------------------------------')
        ds.append(wiki_dataset)
    if args.book_dir is not None:
        book_dataset = creat_book_dataset(args.book_dir, args.tokenizer_file, 512)
        print(book_dataset)
        print(book_dataset[0])
        print('-----------------------------------------')
        ds.append(book_dataset)
    if args.cci2_dir is not None:
        cci2_dataset = create_cci2_dataset(args.cci2_dir, args.tokenizer_file, 512)
        print(cci2_dataset)
        print(cci2_dataset[0])
        print('-----------------------------------------')
        ds.append(cci2_dataset)
    if args.wiki_zh_dir is not None:
        wiki_zh_dataset = create_wiki_zh_dataset(args.wiki_zh_dir, args.tokenizer_file, 512)
        print(wiki_zh_dataset)
        print(wiki_zh_dataset[0])
        print('-----------------------------------------')
        ds.append(wiki_zh_dataset)
    os.makedirs(args.output_dir,exist_ok=True)
    concatenated_dataset = datasets.concatenate_datasets(ds)
    print(concatenated_dataset)
    print(concatenated_dataset[0])
    concatenated_dataset.save_to_disk(args.output_dir)
