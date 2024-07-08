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
"""
export NCCL_SOCKET_IFNAME=eno1
export NODE_RANK=0
export CUDA_HOME=/usr/local/cuda-12.1
export MASTER_ADDR=192.168.1.39
export NCCL_SOCKET_FAMILY=IPv4
export MASTER_PORT=19999
"""
from pytorch_lightning import Trainer
from lightning_utilities.core.rank_zero import rank_zero_info

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
from src.model import RWKV
from src.model_ext import RwkvForSequenceEmbedding, load_ckpt_and_parse_args
from peft_train.data_collators import pad_and_truncated
import torch
from torch.utils.data import DataLoader
def create_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='peft train BiEncoder')
    parser.add_argument('--train_data', type=str,help='parquet dicrectory containing the training data')
    parser.add_argument('--train_lengths',type=int,nargs='+',default=None,help='length of the training data')
    parser.add_argument('--train_batch_sizes', type=int,nargs='+', default=None, help='batch size to train the model')
    parser.add_argument('--model_file', type=str,default='/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth', help='model to be trained,now rwkv5 and rwkv6 are supported')
    parser.add_argument('--output_dir', type=str, default='/media/yueyulin/bigdata/tmp',help='directory to save the trained model')
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs to train the model')
    parser.add_argument('--max_seq_length', type=int, default=512, help='maximum sequence length to train the model')
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
    parser.add_argument('--strategy', type=str, default='deepspeed_stage_2_offload', help='strategy for distributed training')
    parser.add_argument('--my_qa_mask', type=int, default=0)

    parser.add_argument('--train_type', type=str, default='states', help='train type')
    
    #add peft arguments
    parser.add_argument('--lora_type', type=str, default='lora', help='lora type', choices=['lora','adalora'])
    parser.add_argument('--target_modules', type=str, nargs='+',default=None, help='target modules')
    parser.add_argument('--lora_r',type=int,default=8)
    parser.add_argument('--lora_alpha',type=int,default=32)
    parser.add_argument('--lora_dropout',type=float,default=0.1)
    parser.add_argument('--svd_niter',type=int,default=4)
    #add lask peft checkpoint path
    parser.add_argument('--peft_checkpoint',type=str,help='peft checkpoint path',default=None)
    parser.add_argument('--skip_steps',type=int,default=0,help='skip steps in the peft checkpoint')
    parser.add_argument('--pissa_init_file',type=str,help='pissa init file',default=None)

    #add option if use customized train_step
    parser.add_argument('--custom_train_step',type=bool,default=False,help='use customized train step')
    return parser

def configure_args(args):
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = args.micro_bsz * args.accumulate_grad_batches*args.num_devices
    args.my_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.proj_dir is None:
        args.proj_dir = f'{args.output_dir}/{args.my_timestamp}'
    args.wandb = f'{args.wandb}'
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
def load_lora_and_pissa(args, model):
    if args.peft_checkpoint is not None and os.path.exists(args.peft_checkpoint):
        w = torch.load(args.peft_checkpoint)
        info = model.load_state_dict(w,strict=False)
        print(f'load peft checkpoint from {args.peft_checkpoint}, info: {info}')

    if args.peft_checkpoint is not None:
        if args.pissa_init_file is None or not os.path.exists(args.pissa_init_file):
            init_dict = {}

            for name, m in model.named_modules():
                if hasattr(m, "pissa_init") and callable(getattr(m, "pissa_init")):
                    m.pissa_init(args.svd_niter)
                    init_dict[f'{name}.init_lora_A'] = m.lora_A.data
                    init_dict[f'{name}.init_lora_B'] = m.lora_B.data
            save_pth = f'{args.proj_dir}/init_pissa.pth' if args.pissa_init_file is None else args.pissa_init_file
            if not os.path.exists(save_pth):
                print(f"save init pissa to {save_pth}")
                torch.save(init_dict, save_pth)
            else:
                print(f"{save_pth} exists")
        else:
            pissa_init = torch.load(args.pissa_init_file, map_location="cpu")
            print(f"########## Load PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                    m.pissa_load(pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    configure_args(args)
    print(args)
    #print loading data from train_data in red
    import colorama
    print(colorama.Fore.RED + f'loading data from {args.train_data}'+colorama.Fore.RESET)
    # data_path_list = []
    # for data_path in os.listdir(args.train_data):
    #     if data_path.endswith('_dataset'):
    #         data_path_list.append(os.path.join(args.train_data,data_path))
    # ds = read_dataset(data_path_list)
    # print(ds)
    # def data_collator(batch):
    #     input_ids = [b['input_ids'] for b in batch]
    #     labels = [b['labels'] for b in batch]
    #     input_ids = torch.tensor(input_ids)
    #     labels = torch.tensor(labels)
    #     return input_ids, labels
    # train_dataloader = DataLoader(ds,shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True,collate_fn=data_collator)
    # print(train_dataloader)
    if args.train_lengths is not None:
        from data.custom_datasets import read_dataset as read_variable_length_dataset,pad_only_according_data
        ds = read_variable_length_dataset(args.train_data,args.train_lengths)
        length_of_dataset = len(ds)
        sum_of_batches = sum([(ds.cummulative_sizes[i]-(ds.cummulative_sizes[i-1] if i > 0 else 0))//args.train_lengths[i] for i in range(len(ds.cummulative_sizes))])
        print(sum_of_batches)
        batch_size = length_of_dataset // sum_of_batches
        print(batch_size)
        sampler = MyBatchSampler([i for i in range(len(ds))],batch_size,True,ds.cummulative_sizes,args.train_batch_sizes,skipped_batches=args.skip_steps)
        train_dataloader = DataLoader(ds,batch_sampler=sampler,collate_fn=pad_only_according_data)
    else:
        ds = load_from_disk(args.train_data)
        def data_collator(batch):
            input_ids = [b['input_ids'] for b in batch]
            labels = [b['labels'] for b in batch]
            input_ids = torch.tensor(input_ids)
            labels = torch.tensor(labels)
            return input_ids, labels
        train_dataloader = DataLoader(ds,shuffle=True, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True,collate_fn=data_collator)

    args.epoch_steps = len(train_dataloader)//args.num_devices
    
    dev_dataloader = None

    
    w = load_ckpt_and_parse_args(args.model_file,args)
    if args.custom_train_step:
        from torch.nn import functional as F
        tokenizer_file = 'tokenizer/rwkv_vocab_v20230424.txt'
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
        tokenizer = TRIE_TOKENIZER(tokenizer_file)
        import orjson as json
        def is_subsequence(seq, subseq):
            len_subseq = len(subseq)
            windows = seq.unfold(0, len_subseq, 1)
            return (windows == subseq).all(dim=1).any().item()

        def training_step(self, batch, batch_idx):
            args = self.args
            eos_id = 1
            idx, targets = batch
            logits = self(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            #decode logits to ids
            ids = logits.argmax(dim=-1)
            valid_ids_mask = torch.not_equal(targets,-100).int()
            for i in range(targets.size(0)):
                target = targets[i]
                mask = valid_ids_mask[i]
                #get the subsequence of the target tensor
                first_idx = torch.argmax(mask)
                last_idx = torch.eq(target,eos_id).int().argmax()
                decoded_str =tokenizer.decode(target[first_idx:last_idx].detach().tolist())
                generated_ids = ids[i][first_idx:last_idx]
                decoded = json.loads(decoded_str)
                keys = decoded.keys()  
                num_keys = len(keys)
                num_values = 0
                matched_keys = 0
                matched_values = 0
                for key in keys:
                    num_values += len(decoded[key])
                    key_ids = tokenizer.encode(key)
                    key_ids = torch.tensor(key_ids,dtype=torch.long)
                    key_ids = key_ids.to(idx.device)
                    #test if the key is in the generated_ids
                    if is_subsequence(generated_ids,key_ids):
                        matched_keys += 1
                    for value in decoded[key]:
                        value_ids = tokenizer.encode(value)
                        value_ids = torch.tensor(value_ids,dtype=torch.long)
                        value_ids = value_ids.to(idx.device)
                        if is_subsequence(generated_ids,value_ids):
                            matched_values += 1
                #add loss if the key and value are not matched
                key_match_loss = (num_keys - matched_keys) * 3 / num_keys if num_keys != 0 else 0
                value_match_loss = (num_values - matched_values) * 5 / num_values if num_values != 0 else 0
                key_value_match_loss = key_match_loss + value_match_loss
                loss += key_value_match_loss
            return {'loss': loss, 'key_match_loss': key_match_loss, 'value_match_loss': value_match_loss}

        RWKV.training_step = training_step
        print(colorama.Fore.RED +'use customized training step'+colorama.Fore.RESET)
    else:
        print(colorama.Fore.RED +'use default training step'+colorama.Fore.RESET)

    if args.train_type == 'lora': 
        args.lora = True
        args.parts = args.target_modules
        from src.model import LORA_CONFIG
        LORA_CONFIG['r'] = args.lora_r
        LORA_CONFIG['alpha'] = args.lora_alpha
        LORA_CONFIG['dropout'] = args.lora_dropout
        LORA_CONFIG['parts'] = args.parts
        model = RWKV(args)
        print(model)
        inform = model.load_state_dict(w,strict=False)
        print(inform)
        for name, param in model.named_parameters():
            if 'lora_' in name :
                param.requires_grad = True
            else:
                param.requires_grad = False
        """
        Let's use the rwkv's own lora configuration
        model = RWKV(args)
        print(model)
        inform = model.load_state_dict(w,strict=False)
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
        model = inject_adapter_in_model(lora_config,model,adapter_name='sft_lora')
        print(model)
        def print_trainable_params(model):
            #count whole model parameters and print trainable parameters' count and percentage
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(colorama.Fore.GREEN + f'total params: {total_params}, trainable params: {trainable_params}, trainable params percentage: {trainable_params/total_params*100:.2f}%')
        print_trainable_params(model)
        """
    elif args.train_type == 'state':
        args.lora = False
        args.state_tune = True
        if args.target_modules is not None:
            args.parts = args.target_modules
            from src.model import LORA_CONFIG
            LORA_CONFIG['r'] = args.lora_r
            LORA_CONFIG['alpha'] = args.lora_alpha
            LORA_CONFIG['dropout'] = args.lora_dropout
            LORA_CONFIG['parts'] = args.parts
        model = RWKV(args)
        print(model)
        inform = model.load_state_dict(w,strict=False)
        print(inform)
        for name, param in model.named_parameters():
            if 'state' in name :
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.train_type == 'pissa':
        args.lora = True
        args.parts = args.target_modules
        from src.model import LORA_CONFIG
        LORA_CONFIG['r'] = args.lora_r
        LORA_CONFIG['alpha'] = args.lora_alpha
        LORA_CONFIG['dropout'] = args.lora_dropout
        model = RWKV(args)
        print(model)
        inform = model.load_state_dict(w,strict=False)
        print(inform)
        for name, param in model.named_parameters():
            if 'lora_' in name :
                param.requires_grad = True
            else:
                param.requires_grad = False
       
        print(model)
    

    load_lora_and_pissa(args, model)
    #Train the model
    # device = "auto"
    args.skip_steps = 0
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
    if args.train_lengths is not None:
        sampler.set_world_size(trainer.world_size)
        sampler.rank = trainer.global_rank
    trainer.fit(model, 
                train_dataloader,
                dev_dataloader)