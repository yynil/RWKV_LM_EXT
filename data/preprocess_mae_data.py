import glob
import os
os.environ['HF_DATASETS_CACHE'] = '/media/yueyulin/data_4t/cache'
import datasets
import random
import functools
tokenizer = None
ht = None
def create_cci2_dataset(cci2_dir,
                            tokenizer_file,
                            max_seq_length: int,
                            short_seq_prob: float = 0.0):
    parquet_files = glob.glob(os.path.join(cci2_dir, '*.parquet'))
    print(f'Found {len(parquet_files)} parquet files in {cci2_dir}')
    ds = datasets.load_dataset('parquet', data_files=parquet_files)['train']
    print(f'Loaded dataset with {len(ds)} samples')
    print(ds)
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
        global ht
        if ht is None:
            from harvesttext import HarvestText
            ht = HarvestText()
        sentences = ht.cut_sentences(examples["content"])
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
    wiki = ds.map(sentence_wiki, num_proc=16, remove_columns=["id", "content"])
    tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=16, batched=True, remove_columns=["sentences"])
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
    processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=16, batched=True, remove_columns=tokenized_wiki.column_names)
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
    parser.add_argument('--output_dir', type=str,default='/media/yueyulin/data_4t/data/zh_mae_dataset')
    parser.add_argument('--cci2_dir',type=str,default='/media/yueyulin/data_4t/data/CCI2-Data')
    parser.add_argument('--tokenizer_file', type=str,default='/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt')
    args = parser.parse_args()
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    print(f'appended {parent_dir} to sys.path')
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
    os.makedirs(args.output_dir,exist_ok=True)
    concatenated_dataset = datasets.concatenate_datasets(ds)
    print(concatenated_dataset)
    print(concatenated_dataset[0])
    concatenated_dataset.save_to_disk(args.output_dir)