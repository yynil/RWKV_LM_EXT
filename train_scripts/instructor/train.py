import os
import sys

from pytorch_lightning import Trainer

parent_parent_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(parent_parent_dir)
print('add ',parent_parent_dir,' to sys path')
#HF_ENDPOINT=https://hf-mirror.com
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print('add ',parent_parent_dir,' to sys path')
#HF_ENDPOINT=https://hf-mirror.com
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
os.environ['RWKV_TRAIN_TYPE'] = ''
os.environ["WKV"] = ''
import orjson as json
import torch
import random
from datasets import Dataset,DatasetDict
from src.model_ext import load_ckpt_and_parse_args,RwkvStatesForSequenceEmbedding,RwkvInstructorForSequenceEmbedding
import gc
from peft_train.Callbacks import TrainerCallback
instructor_tokenizer = None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Train an instructor embedding model")
    parser.add_argument("--tokenizer", type=str,default="/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt")
    parser.add_argument("--output_dir", type=str, default="/media/yueyulin/KINGSTON/models/instructor_embedding")
    parser.add_argument("--training_data", type=str,default="/media/yueyulin/data_4t/data/instructor_embeddings/medi-data/medi-data/medi-data.json")
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--pretrained_model', type=str, default="/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth")
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs to train the model')
    parser.add_argument('--num_devices', type=int, default = 1,help='number of devices to train the model')
    parser.add_argument('--my_pos_emb', type=int, default=0, help='default 0, which is the position embedding in the model')
    parser.add_argument('--pre_ffn', type=int, default=0, help='default 0, which is the pre feed forward network in the model')
    parser.add_argument('--head_size_divisor', type=int, default=8, help='head size divisor in the model')
    parser.add_argument('--ctx_len', type=int, default=4096, help='context length in the model')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate in the model')
    parser.add_argument('--head_qk', type=int, default=0, help='head query key in the model')
    parser.add_argument('--grad_cp', type=int, default=0, help='gradient checkpoint in the model')
    parser.add_argument('--save_per_batches', type=int, default=10000, help='number of batches to save the model')
    parser.add_argument('--my_exit', type=int, default=300, help='exit condition in the model')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay in the model')
    parser.add_argument('--lr_init', type=float, default=3e-4, help='initial learning rate in the model')
    parser.add_argument('--lr_final', type=float, default=1e-5, help='final learning rate in the model')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter in the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 parameter in the Adam optimizer')
    parser.add_argument('--layerwise_lr', type=float, nargs='+', default=1, help='layerwise learning rate in the model')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='epsilon parameter in the Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=50, help='warmup steps in the model')
    parser.add_argument('--tiny_att_dim', type=int, default=0, help='tiny attention dimension in the model')
    parser.add_argument('--epoch_begin', type=int, default=0, help='beginning epoch for the training')
    parser.add_argument('--epoch_count', type=int, default=150, help='total number of epochs for the training')
    parser.add_argument('--epoch_save', type=int, default=1, help='number of epochs after which the model is saved')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum number of epochs for the training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='number of epochs after which the validation is checked')
    parser.add_argument('--val_check_interval', type=int, default=100, help='number of epochs after which the validation is checked')
    parser.add_argument('--num_sanity_val_steps', type=int, default=0, help='number of validation steps for sanity check at the beginning of training')
    parser.add_argument('--log_every_n_steps', type=int, default=1000, help='number of steps after which the training progress will be logged')
    parser.add_argument('--enable_checkpointing', type=bool, default=False, help='flag to enable checkpointing')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='number of batches to accumulate before performing a backward/update pass')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='maximum gradient norm')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--micro_bsz', type=int,default=4, help='micro batch size for training')
    parser.add_argument('--real_bsz', type=int, help='real batch size for training')
    parser.add_argument('--my_pile_stage', type=int, default=0, help='pile stage in the model')
    parser.add_argument('--my_pile_edecay', type=float, default=0, help='pile exponential decay in the model')
    parser.add_argument('--weight_decay_final', type=float, default=-1, help='final weight decay in the model')
    parser.add_argument('--proj_dir', type=str, help='project directory to save the model and logs')
    parser.add_argument('--eval_every_steps', type=int, default=100, help='number of steps after which the model is evaluated')
    parser.add_argument('--wandb', type=str, default='instructor', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='embedder', help='run name for wandb logging')
    parser.add_argument('--strategy', type=str, default='deepspeed_stage_2_offload', help='strategy for distributed training')
    parser.add_argument('--my_qa_mask', type=int, default=0)

    parser.add_argument('--chunk_ctx',type=int,default=256,help='chunk context length')
    parser.add_argument('--cl_temperature',type=float,default=0.1,help='chunk level temperature')
    parser.add_argument('--skip_steps',type=int,default=0,help='skip steps in the peft checkpoint')
    parser.add_argument('--max_ctx',type=int,default=512,help='max ctx length')
    parser.add_argument('--cache_dir',type=str,default='/tmp/cache',help='cache directory for the model')
    args = parser.parse_args()
    args.real_bsz = max(args.micro_bsz,
                            args.micro_bsz * torch.cuda.device_count())
    cache_dir = f'{args.cache_dir}/max_ctx_{args.max_ctx}_real_bsz_{args.real_bsz}'  
    os.makedirs(cache_dir,exist_ok=True)
    os.makedirs(args.output_dir,exist_ok=True)
    from datetime import datetime
    args.my_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    args.run_name = f'{args.run_name}-{args.my_timestamp}'
    args.betas = [args.beta1,args.beta2]
    args.trainable_dir_output = f'{args.output_dir}/{args.my_timestamp}'

    w = load_ckpt_and_parse_args(args.pretrained_model,args)
    print(args)
    # model = RwkvStatesForSequenceEmbedding(args)
    model = RwkvInstructorForSequenceEmbedding(args)
    print(model)
    info = model.load_state_dict(w,strict=False)
    print(info)
    del w
    gc.collect()
    #check if cache_dir dataset exists
    train_dataset = None
    try:
        from datasets import load_from_disk
        train_dataset = load_from_disk(cache_dir)
        print(f'Loaded cache_dir dataset from {cache_dir}')
    except Exception as ex:
        print(ex)
        print(f'Error in loading {cache_dir} dataset')
    if train_dataset is None:
        with open(args.training_data) as f:
            train_examples_raw = json.loads(f.read())
        print(f"Loaded {len(train_examples_raw)} training examples")
        total_train_n = len(train_examples_raw)
        def get_examples_raw(old_examples_raw, total_n, real_batch_size):
            examples_raw = []
            for idx in range(0, total_n, real_batch_size):
                local_task_name = old_examples_raw[idx]['task_name']
                cur_batch = []
                include_batch = True
                for idx1 in range(idx, min(idx + real_batch_size, total_n)):
                    if not old_examples_raw[idx1]['task_name'] == local_task_name:
                        print(f'one batch in task {old_examples_raw[idx1]["task_name"]} is skipped')
                        include_batch = False
                        break
                    else:
                        cur_batch.append(old_examples_raw[idx1])
                if include_batch and len(cur_batch) == real_batch_size:
                    examples_raw.append(cur_batch)
            return examples_raw
        
        train_examples_raw = get_examples_raw(train_examples_raw, total_train_n, args.real_bsz)
        random.shuffle(train_examples_raw)
        train_examples_raw_batch = train_examples_raw
        train_examples_raw = []

        for b in train_examples_raw_batch:
            train_examples_raw += b
        print(f'There are {len(train_examples_raw)} pairs to train in total.')
        print(f'{len(train_examples_raw[0])}')
        print(train_examples_raw[0])

        def get_dataset(examples_raw):
            examples = {'query':[],'pos':[],'neg':[],'task_id':[]}
            task_name_map = {}
            total_num = len(examples_raw)
            task_count = 0
            for i in range(total_num):
                cur_e = examples_raw[i]
                for k in ['query','pos','neg']:
                    cur_e[k][-1] = str(cur_e[k][-1])
                    assert cur_e[k][0].startswith('Represent ') or cur_e[k][0]==''
                    examples[k].append("".join(cur_e[k]))
                if not cur_e['task_name'] in task_name_map:
                    task_name_map[cur_e['task_name']] = task_count
                    task_count += 1
                examples['task_id'].append(task_name_map[cur_e['task_name']])
            return examples

        train_raw_datasets = DatasetDict({'train':Dataset.from_dict(get_dataset(train_examples_raw))})

        column_names = train_raw_datasets["train"].column_names
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

        
        def preprocess_function(examples):
            global instructor_tokenizer
            all_tokenized = {'query_input_ids':[],'pos_input_ids':[],'neg_input_ids':[]}
            if instructor_tokenizer is None:
                instructor_tokenizer = TRIE_TOKENIZER(args.tokenizer)
            for key in ['query','pos','neg']:
                input_ids = [
                    instructor_tokenizer.encode(example) 
                        for example 
                            in examples[key]]
                all_tokenized[f'{key}_input_ids'] = input_ids
            all_tokenized['task_id'] = examples['task_id']
            return all_tokenized

        train_dataset = train_raw_datasets["train"]
        train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=4,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )
        print(train_dataset)
        print(train_dataset[0])
        #save it to cache_dir
        print(f'Saving train_dataset to {cache_dir}')
        train_dataset.save_to_disk(cache_dir)

    from torch.utils.data import DataLoader
    pad_id = 0
    eos_id = 1
    MAX = 511
    def pad(batch):
        data = []
        for x in batch:
            x.extend([pad_id]*(MAX-len(x)))
            if len(x) > MAX:
                x = x[:MAX]
            x.append(eos_id)
            data.append(x)
        return data
    def padding_fn(batch):
        query_input_ids = pad([b['query_input_ids'] for b in batch])
        pos_input_ids = pad([b['pos_input_ids'] for b in batch])
        neg_input_ids = pad([b['neg_input_ids'] for b in batch])
        return {'query_input_ids':torch.tensor(query_input_ids,dtype=torch.long),
                'pos_input_ids':torch.tensor(pos_input_ids,dtype=torch.long),
                'neg_input_ids':torch.tensor(neg_input_ids,dtype=torch.long),}
    data_loader = DataLoader(train_dataset, batch_size=args.micro_bsz, shuffle=False,collate_fn=padding_fn)
    args.epoch_steps = len(data_loader)//args.num_devices

    trainer = Trainer(accelerator="auto",
                      strategy=args.strategy,
                      devices=args.num_devices,
                      num_nodes=args.num_nodes,
                      precision=precision,
                      logger=False,
                      callbacks=[TrainerCallback(args)],
                      max_epochs=args.max_epochs,
                      check_val_every_n_epoch=args.check_val_every_n_epoch,
                      num_sanity_val_steps=args.num_sanity_val_steps,
                      log_every_n_steps=args.log_every_n_steps,
                      enable_checkpointing=args.enable_checkpointing,
                      accumulate_grad_batches=args.accumulate_grad_batches,
                      gradient_clip_val=args.gradient_clip_val,
                      val_check_interval=args.val_check_interval,
                      use_distributed_sampler=False)

    
    print("Current device rank: ", trainer.global_rank)
    print("Total number of devices: ", trainer.world_size)
    trainer.fit(model, 
                data_loader)