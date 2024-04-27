#peft train BiEncoder
#This script accept the following arguments:
#  --train_data TRAIN_DATA which is a parquet dicrectory containing the training data
#  --model MODEL which is the model to be trained,now rwkv5 and rwkv6 are supported
#  --output_dir OUTPUT_DIR which is the directory to save the trained model
#  --num_epochs NUM_EPOCHS which is the number of epochs to train the model
#  --batch_size BATCH_SIZE which is the batch size to train the model
#  --learning_rate LEARNING_RATE which is the learning rate to train the model
#  --warmup_steps WARMUP_STEPS which is the warmup steps to train the model
#  --max_seq_length MAX_SEQ_LENGTH which is the maximum sequence length to train the model
#  --add_mlp ADD_MLP which is the flag to add mlp to the model
#  --mlp_dim MLP_DIM which is the dimension of the mlp
#  --pooling_type POOLING_TYPE which is the pooling type to train the model, the candidates are ['weightedmean','lasttoken']
#  --is_in_batch_negative IS_IN_BATCH_NEGATIVE which is the flag to use in batch negative sampling
#  --num_devices NUM_DEVICES which is the number of devices to train the model
#  --my_pos_emb MY_POS_EMB default 0, which is the position embedding in the model
#  --pre_ffn PRE_FFN default 0, which is the pre feed forward network in the model
#  --head_size_divisor HEAD_SIZE_DIVISOR which is the head size divisor in the model
#  --ctx_len CTX_LEN which is the context length in the model
#  --dropout DROPOUT which is the dropout rate in the model
#  --head_qk HEAD_QK which is the head query key in the model
#  --grad_cp GRAD_CP which is the gradient checkpoint in the model
#  --save_per_batches SAVE_PER_BATCHES which is the number of batches to save the model
#  --my_exit MY_EXIT which is the exit condition in the model
#  --weight_decay WEIGHT_DECAY which is the weight decay in the model
#  --lr_init LR_INIT which is the initial learning rate in the model
#  --lr_final LR_FINAL which is the final learning rate in the model
#  --beta1 BETA1 which is the beta1 parameter in the Adam optimizer
#  --beta2 BETA2 which is the beta2 parameter in the Adam optimizer
#  --betas BETAS which is the betas parameter in the Adam optimizer
#  --layerwise_lr LAYERWISE_LR which is the layerwise learning rate in the model
#  --my_pile_stage MY_PILE_STAGE which is the pile stage in the model
#  --adam_eps ADAM_EPS which is the epsilon parameter in the Adam optimizer
#  --warmup_steps WARMUP_STEPS which is the warmup steps in the model
#  --tiny_att_dim TINY_ATT_DIM which is the tiny attention dimension in the model
#  --epoch_begin EPOCH_BEGIN which is the beginning epoch for the training
#  --epoch_count EPOCH_COUNT which is the total number of epochs for the training
#  --epoch_save EPOCH_SAVE which is the number of epochs after which the model is saved
#  --max_epochs MAX_EPOCHS which is the maximum number of epochs for the training
#  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH which is the number of epochs after which the validation is checked
#  --num_sanity_val_steps NUM_SANITY_VAL_STEPS which is the number of validation steps to perform for sanity check at the beginning of training
#  --log_every_n_steps LOG_EVERY_N_STEPS which is the number of steps after which the training progress will be logged
#  --enable_checkpointing ENABLE_CHECKPOINTING which is a flag to enable checkpointing
#  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES which is the number of batches to accumulate before performing a backward/update pass
#  --gradient_clip_val GRADIENT_CLIP_VAL which is the maximum gradient norm
#  --num_nodes NUM_NODES which is the number of nodes for distributed training
#  --devices DEVICES which is the number of devices for distributed training
#  --micro_bsz MICRO_BSZ which is the micro batch size for training
#  --real_bsz REAL_BSZ which is the real batch size for training
#  --my_pile_stage MY_PILE_STAGE which is the pile stage in the model
#  --my_pile_edecay MY_PILE_EDECAY which is the pile exponential decay in the model
#  --weight_decay_final WEIGHT_DECAY_FINAL which is the final weight decay in the model
#  --proj_dir PROJ_DIR which is the project directory to save the model and logs
#  --eval_every_steps EVAL_EVERY_STEPS which is the number of steps after which the model is evaluated
#  --my_timestamp MY_TIMESTAMP which is the timestamp of the training
#  --wandb WANDb which is the wandb project name
#  --run_name RUN_NAME which is the run name for wandb logging
from datetime import datetime
from functools import partial
from datasets import Dataset,load_from_disk,concatenate_datasets
import os
import sys

from pytorch_lightning import Trainer

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_parent_dir)
print('add ',parent_parent_dir,' to sys path')
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'

from peft_train.Callbacks import TrainerCallback
from src.model_ext import load_ckpt_and_parse_args
from src.model_bi import RWKV
from src.model_bi import RwkvForSequenceEmbedding
from peft_train.data_collators import pad_and_truncated
import torch
from torch.utils.data import DataLoader
def create_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='peft train BiEncoder')
    parser.add_argument('--train_data', type=str,help='parquet dicrectory containing the training data',default='/media/yueyulin/data_4t/datasets/zh_wiki_tokenized_chunked_255')
    parser.add_argument('--train_lengths',type=int,nargs='+',default=[128,256,512,1024,2048,4096],help='length of the training data')
    parser.add_argument('--train_batch_sizes', type=int,nargs='+', default=[32,16,8,4,2,1], help='batch size to train the model')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size to train the model')
    parser.add_argument('--dev_data', type=str,nargs='+' ,help='parquet dicrectory containing the dev data')
    parser.add_argument('--model_file', type=str,default='/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth', help='model to be trained,now rwkv5 and rwkv6 are supported')
    parser.add_argument('--output_dir', type=str, default='/media/yueyulin/KINGSTON/models/rwkv6/tmp',help='directory to save the trained model')
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs to train the model')
    parser.add_argument('--max_seq_length', type=int, default=128, help='maximum sequence length to train the model')
    parser.add_argument('--add_mlp', action='store_true', help='flag to add mlp to the model')
    parser.add_argument('--mlp_dim', type=int,default=1024, help='dimension of the mlp')
    parser.add_argument('--pooling_type', type=str, default='weightedmean',help='pooling type to train the model, the candidates are [\'weightedmean\',\'lasttoken\']')
    parser.add_argument('--is_in_batch_negative', action='store_true', help='flag to use in batch negative sampling')
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
    parser.add_argument('--wandb', type=str, default='peft', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='peft_bi_encoder_trainer', help='run name for wandb logging')
    parser.add_argument('--is_unsupervised', action='store_true', help='flag to use unsupervised training',default=False)

    #add peft arguments
    parser.add_argument('--lora_type', type=str, default='lora', help='lora type', choices=['lora','adalora'])
    parser.add_argument('--target_modules', type=str, nargs='+',default=['ffn.key','ffn.value','ffn.receptance'], help='target modules')
    parser.add_argument('--lora_r',type=int,default=8)
    parser.add_argument('--lora_alpha',type=int,default=32)
    parser.add_argument('--lora_dropout',type=float,default=0.1)

    #add lask peft checkpoint path
    parser.add_argument('--peft_checkpoint',type=str,help='peft checkpoint path',default=None)
    parser.add_argument('--skip_steps',type=int,default=0,help='skip steps in the peft checkpoint')
    return parser

def configure_args(args):
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = args.micro_bsz * args.accumulate_grad_batches*args.num_devices
    args.my_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.proj_dir is None:
        args.proj_dir = f'{args.output_dir}/{args.my_timestamp}'
    args.wandb = f'{args.wandb}-{args.my_timestamp}'
    args.run_name = f'{args.run_name}-{args.my_timestamp}'

    args.trainable_dir_output = os.path.join(args.proj_dir, "trainable_model")
    os.makedirs(args.trainable_dir_output, exist_ok=True)

def read_dataset(data_path_list):
    ds = []
    for data_path in data_path_list:
        ds.append(load_from_disk(data_path))
    return concatenate_datasets(ds)
from data.custom_datasets import read_dataset as read_variable_length_dataset
from data.custom_datasets import MyBatchSampler,pad_and_truncated_according_data
if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    configure_args(args)
    print(args)
    #print loading data from train_data in red
    import colorama
    print(colorama.Fore.RED + f'loading data from {args.train_data}')
    if args.is_unsupervised:
        from datasets import load_from_disk
        ds = load_from_disk(args.train_data)
        def pad_and_truncated(features, max_len, pad_token_id=0,eos_token_id=1):
            query_ids = [feature+[eos_token_id] for feature in features['input_ids']]
            query_ids = [q[:max_len] for q in query_ids]
            query_ids = [q+[pad_token_id]*(max_len-len(q)) for q in query_ids]
            #clone the query_ids as positive_ids
            positive_ids = [q for q in query_ids]
            return {'query':torch.tensor(query_ids,dtype=torch.long),
                    'positive':torch.tensor(positive_ids,dtype=torch.long)}
        max_len = 256
        from functools import partial
        pad_and_truncated_partial = partial(pad_and_truncated,max_len=max_len)
        ds = ds.map(pad_and_truncated_partial,batched=True,num_proc=8,remove_columns=['input_ids'])
        print(ds)
        def collate_fn(batch):
            query = [b['query'] for b in batch]
            positive = [b['positive'] for b in batch]
            return {'query':torch.tensor(query,dtype=torch.long),'positive':torch.tensor(positive,dtype=torch.long)}
        train_dataloader = DataLoader(ds['train'],batch_size=args.batch_size,collate_fn=collate_fn)
    else:
        ds = read_variable_length_dataset(args.train_data,args.train_lengths)
        length_of_dataset = len(ds)
        sum_of_batches = sum([(ds.cummulative_sizes[i]-(ds.cummulative_sizes[i-1] if i > 0 else 0))//args.train_lengths[i] for i in range(len(ds.cummulative_sizes))])
        print(sum_of_batches)
        batch_size = length_of_dataset // sum_of_batches
        print(batch_size)
        sampler = MyBatchSampler([i for i in range(len(ds))],batch_size,True,ds.cummulative_sizes,args.train_batch_sizes)
        train_dataloader = DataLoader(ds,batch_sampler=sampler,collate_fn=pad_and_truncated_according_data)
    
    total_steps = len(train_dataloader)
    if args.skip_steps > 0:
        import itertools
        print(colorama.Fore.RED + f'skip {args.skip_steps} steps'+colorama.Style.RESET_ALL)
        train_dataloader = itertools.islice(train_dataloader,args.skip_steps,None)
        args.epoch_steps = (total_steps - args.skip_steps) // args.num_devices
    else:
        args.epoch_steps = len(train_dataloader)//args.num_devices
    collator = partial(pad_and_truncated,max_len=args.max_seq_length)
    #print loading dev data from dev_data in red
    # print(colorama.Fore.RED + f'loading dev data from {args.dev_data}')
    # dev_ds = read_dataset(args.dev_data)
    # dev_dataloader = DataLoader(dev_ds,batch_size=args.micro_bsz,shuffle=False,pin_memory=True,num_workers=4,collate_fn=collator)

    
    w = load_ckpt_and_parse_args(args.model_file,args)

    rwkv_base_model = RWKV(args)
    print(rwkv_base_model)
    inform = rwkv_base_model.load_state_dict(w)
    print(inform)


    #Configure the peft configuration to inject 
    lora_config = None
    if args.lora_type == 'lora':
        from peft import LoraConfig
        lora_config = LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,target_modules=args.target_modules,lora_dropout=args.lora_dropout)
    elif args.lora_type == 'adalora':
        from peft import AdaLoraConfig
        lora_config = AdaLoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,target_modules=args.target_modules,lora_dropout=args.lora_dropout)

    #Inject the lora configuration to the model
    from peft import inject_adapter_in_model
    rwkv_base_model = inject_adapter_in_model(lora_config,rwkv_base_model,adapter_name='embedding_lora')
    print(rwkv_base_model)
    if args.peft_checkpoint is not None:
        #load the peft checkpoint
        w = torch.load(args.peft_checkpoint,map_location='cpu')
        infom = rwkv_base_model.load_state_dict(w,strict=False)
        print(colorama.Fore.RED + f'load peft checkpoint from {args.peft_checkpoint} with {infom}'+colorama.Style.RESET_ALL)

    def print_trainable_params(model):
        #count whole model parameters and print trainable parameters' count and percentage
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(colorama.Fore.GREEN + f'total params: {total_params}, trainable params: {trainable_params}, trainable params percentage: {trainable_params/total_params*100:.2f}%')
    print_trainable_params(rwkv_base_model)


    embedding_model = RwkvForSequenceEmbedding(rwkv_base_model,
                                               pooling_type=args.pooling_type,
                                               add_mlp=args.add_mlp,
                                               is_in_batch_negative=args.is_in_batch_negative,
                                               output_dim=args.mlp_dim)
    print(embedding_model)

    #Train the model
    device = "cuda"
    trainer = Trainer(accelerator=device,
                      strategy="deepspeed_stage_2_offload",
                      devices='auto',
                      num_nodes=1,
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
    if not args.is_unsupervised:
        sampler.set_world_size(trainer.world_size)
        sampler.rank = trainer.global_rank

    trainer.fit(embedding_model, 
                train_dataloader,
                # dev_dataloader
                )