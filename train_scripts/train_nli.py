import types
from sentence_transformers import models, util, evaluation, losses
import logging
import sys
import os

parent_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_parent_dir)
print('Add path', parent_parent_dir)
from train_scripts.customer_datasets import ListDataset
from train_scripts.evaluators import EmbeddingSimilarityEvaluator
import gzip
from torch.utils.data import DataLoader
from datetime import datetime
import argparse
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
from src.model import RWKV
from src.model_ext import load_ckpt_and_parse_args,RwkvForSequenceEmbedding

from torch.utils.data import DataLoader
import torch
import math
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import time

import csv
import random
def save_trainable_parameters(model, trainable_dir_output, model_filename):
    print(f"save trainable parameters to {trainable_dir_output} pretrained from {model_filename}")
    # 创建保存目录
    os.makedirs(trainable_dir_output, exist_ok=True)
    
    # 获取可训练的参数
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    # 判断是否有可训练的参数
    if len(trainable_params) == 0:
        print("没有可训练的参数")
        return

    # 保存可训练的参数
    save_filename = os.path.basename(model_filename) + '.pth'
    save_path = os.path.join(trainable_dir_output, save_filename)
    state_dict = {name: param.data for name, param in model.named_parameters() if param.requires_grad}
    torch.save(state_dict, save_path)
    print(f"save trainable parameters to {save_path}")


class NLI_Callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay



        # rank_zero_info(f"{real_step} {lr}")

        if trainer.is_global_zero:
            if  trainer.global_step == 0: # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # logging   
            if real_step % args.log_every_n_steps == 0:
                print(f'saving trainable to {args.trainable_dir_output}')
                print(f"{real_step} {trainer.my_loss:.6f} {math.exp(trainer.my_loss):.4f}  {trainer.current_epoch}, now saving...")
                output_dir = f"{args.trainable_dir_output}/epoch_{trainer.current_epoch}_step_{real_step}"
                save_trainable_parameters(pl_module, output_dir, args.model_file)
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss,   "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))
            if args.eval_every_steps > 0 and real_step % args.eval_every_steps == 0:
                print(f'evaluating at step {real_step}')
                pl_module.eval()
                from sentence_transformers.evaluation import  SimilarityFunction
                evaluator = EmbeddingSimilarityEvaluator(data_loader=args.val_dataloader,main_similarity=SimilarityFunction.COSINE,precision='float32')
                eval_value = evaluator(pl_module,output_path=args.trainable_dir_output,epoch=trainer.current_epoch,steps=real_step)
                print(f"eval_value {eval_value} at step {real_step}")
                if len(args.wandb) > 0:
                    trainer.my_wandb.log({"eval_value": eval_value}, step=int(real_step))
                pl_module.train()
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f}  {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)
            output_dir = f"{args.trainable_dir_output}/epoch_{trainer.current_epoch}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_trainable_parameters(pl_module, output_dir, args.model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_rwkv_model', type=str, default='/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth', help='base rwkv model for training')
    parser.add_argument('--nil_file', type=str, default='/media/yueyulin/bigdata/data/nli/AllNLI.tsv', help='Path to the training data')
    parser.add_argument('--sts_file', type=str, default='/media/yueyulin/bigdata/data/stsbenchmark/stsbenchmark.tsv', help='Path to the training data')
    parser.add_argument('--output_dir', type=str, default='/media/yueyulin/bigdata/output/wiki1m', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_seq_length', type=int, default=33, help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=0.05, help='Temperature parameter for the softmax')
    parser.add_argument('--num_devices',type=int, default=1, help='Number of devices for training')
    args = parser.parse_args()

    nil_file = args.nil_file
    sts_file = args.sts_file

    train_data = {}
    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)
    #add nil_file's data to train_data
    with open(nil_file, 'r',encoding='UTF-8') as f:
        reader = csv.DictReader(f, delimiter='\t',quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split']=='train':
                sent1 = row['sentence1'].strip()
                sent2 = row['sentence2'].strip()
                label = row['label'].strip()
                add_to_samples(sent1, sent2, label)
                add_to_samples(sent2, sent1, label)

    train_samples = []
    for sent1,others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append({'query':sent1,'positive':random.choice(list(others['entailment'])),'negative':random.choice(list(others['contradiction']))})
            train_samples.append({'query':random.choice(list(others['entailment'])),'positive':sent1,'negative':random.choice(list(others['contradiction']))})
    del train_data
    #remove duplicated data in train_samples
    import json
    train_samples = [json.loads(item) for item in set(json.dumps(d, sort_keys=True) for d in train_samples)]

    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer_file = os.path.join(parent_parent_dir,'tokenizer','rwkv_vocab_v20230424.txt')
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    

    # define a data collator to tokenize and pad the sentences to the same length
    pad_id =0
    embedding_id = 1
    max_seq_length = args.max_seq_length

    def data_collator(features):
        query_str = [feature['query'] for feature in features]
        positive_str = [feature['positive'] for feature in features]
        negative_str = [feature['negative'] for feature in features]
        query_ids = [tokenizer.encode(query)[0:max_seq_length-1]+[embedding_id] for query in query_str]
        positive_ids = [tokenizer.encode(positive)[0:max_seq_length-1]+[embedding_id] for positive in positive_str]
        negative_ids = [tokenizer.encode(negative)[0:max_seq_length-1]+[embedding_id] for negative in negative_str]
        # pad the sequences to the same length
        query_ids = [ids + [pad_id]*(max_seq_length-len(ids)) for ids in query_ids]
        positive_ids = [ids + [pad_id]*(max_seq_length-len(ids)) for ids in positive_ids]
        negative_ids = [ids + [pad_id]*(max_seq_length-len(ids)) for ids in negative_ids]
        return {'query':torch.tensor(query_ids,dtype=torch.long),'positive':torch.tensor(positive_ids,dtype=torch.long),'negative':torch.tensor(negative_ids,dtype=torch.long)}

    train_ds = ListDataset(train_samples)
    print(len(train_ds))
    print(train_ds[0])
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=args.train_batch_size, collate_fn=data_collator,prefetch_factor=20,num_workers=4)
    print(train_dataloader)

    #load eval data
    eval_data = []
    with open(sts_file, 'r',encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                eval_data.append(row)

    def eval_data_collator(features):
        sentence1_str = [feature['sentence1'] for feature in features]
        sentence2_str = [feature['sentence2'] for feature in features]
        scores = [float(feature['score'])/5.0 for feature in features]
        sentence1_ids = [tokenizer.encode(sentence)[0:max_seq_length-1]+[embedding_id] for sentence in sentence1_str]
        sentence2_ids = [tokenizer.encode(sentence)[0:max_seq_length-1]+[embedding_id] for sentence in sentence2_str]
        # pad the sequences to the same length
        sentence1_ids = [ids + [pad_id]*(max_seq_length-len(ids)) for ids in sentence1_ids]
        sentence2_ids = [ids + [pad_id]*(max_seq_length-len(ids)) for ids in sentence2_ids]
        return {'sentence1':torch.tensor(sentence1_ids,dtype=torch.long),'sentence2':torch.tensor(sentence2_ids,dtype=torch.long),'scores':torch.tensor(scores,dtype=torch.float)}
    
    eval_ds = ListDataset(eval_data)
    print(eval_ds[0])
    eval_dl = DataLoader(eval_ds, batch_size=16, collate_fn=eval_data_collator)

    print('load base model from ',args.base_rwkv_model)
    rwkv_args = argparse.Namespace()
    w = load_ckpt_and_parse_args(args.base_rwkv_model,rwkv_args)
    print(rwkv_args)
    rwkv_args.my_pos_emb = 0
    rwkv_args.pre_ffn = 0
    rwkv_args.head_size_divisor = 8
    rwkv_args.ctx_len = 4096
    rwkv_args.dropout = 0.1
    rwkv_args.head_qk = 0
    rwkv_args.grad_cp = 0
    rwkv_args.save_per_batches = 10000
    rwkv_args.my_exit = 300
    rwkv_args.weight_decay = 0.001
    rwkv_args.lr_init = 3e-4
    rwkv_args.lr_final = 1e-5
    rwkv_args.beta1 = 0.9
    rwkv_args.beta2 = 0.99
    rwkv_args.betas = (0.9, 0.99)
    rwkv_args.layerwise_lr = 1
    rwkv_args.my_pile_stage = 1
    rwkv_args.adam_eps = 1e-8
    rwkv_args.warmup_steps = 50
    rwkv_args.tiny_att_dim = 0
    rwkv_args.model_file = args.base_rwkv_model

    rwkv_args.epoch_begin = 0
    rwkv_args.epoch_count = 150
    rwkv_args.epoch_save = 1
    rwkv_args.max_epochs = 150
    rwkv_args.check_val_every_n_epoch = 1
    rwkv_args.num_sanity_val_steps = 0
    rwkv_args.log_every_n_steps = 1000
    rwkv_args.enable_checkpointing = False
    rwkv_args.accumulate_grad_batches = 1
    rwkv_args.gradient_clip_val = 1.0
    rwkv_args.num_nodes = 1
    rwkv_args.devices = args.num_devices
    rwkv_args.micro_bsz = args.train_batch_size
    rwkv_args.real_bsz = int(rwkv_args.num_nodes) * int(rwkv_args.devices) * rwkv_args.micro_bsz
    rwkv_args.epoch_steps = len(train_ds) // rwkv_args.real_bsz
    rwkv_args.my_pile_stage = 0
    rwkv_args.my_pile_edecay = 0
    rwkv_args.weight_decay_final = -1
    rwkv_args.proj_dir = args.output_dir
    rwkv_args.eval_every_steps = 100
    os.makedirs(rwkv_args.proj_dir, exist_ok=True)
    from datetime import datetime
    rwkv_args.my_timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    rwkv_args.wandb = 'nli'
    rwkv_args.run_name = 'nli_trainer'
    rwkv_args.trainable_dir_output = os.path.join(rwkv_args.proj_dir, "trainable_model")
    rwkv_args.val_dataloader = eval_dl
    print(rwkv_args)
    rwkv_base_model = RWKV(rwkv_args)
    print(rwkv_base_model)
    inform = rwkv_base_model.load_state_dict(w)
    print(inform)

    embedding_model = RwkvForSequenceEmbedding(rwkv_base_model)
    print(embedding_model)

    
    # replace the train_step in the model\

    device = "cuda"
    trainer = Trainer(accelerator=device,
                      strategy="deepspeed_stage_2_offload",
                      devices='auto',
                      num_nodes=1,
                      precision=precision,
                      logger=False,
                      callbacks=[NLI_Callback(rwkv_args)],
                      max_epochs=rwkv_args.max_epochs,
                      check_val_every_n_epoch=rwkv_args.check_val_every_n_epoch,
                      num_sanity_val_steps=rwkv_args.num_sanity_val_steps,
                      log_every_n_steps=rwkv_args.log_every_n_steps,
                      enable_checkpointing=rwkv_args.enable_checkpointing,
                      accumulate_grad_batches=rwkv_args.accumulate_grad_batches,
                      gradient_clip_val=rwkv_args.gradient_clip_val)

    

    trainer.fit(embedding_model, train_dataloader)