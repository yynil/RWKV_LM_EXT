import torch
import torch.nn as nn

import deepspeed
import math
import gc
import os
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch import nn
from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam
from sentence_transformers.util import pairwise_cos_sim
import traceback
from src.infctx_module import BlockState, BlockStateList, TimeMixState
# from src.model import RUN_CUDA_RWKV6_STATE, RWKV_CMix_x060_infctx, RWKV_Tmix_x060_infctx
from src.model import Block,  RWKV_ChannelMix
from torch.utils.checkpoint import checkpoint as torch_checkpoint
MyModule = nn.Module
def __nop(ob):
    return ob
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


def load_embedding_ckpt_and_parse_args(ckpt_file, args):
    try:
        with torch.no_grad():
            w = torch.load(ckpt_file, map_location='cpu') # load model to CPU first
            args.MODEL_NAME = ckpt_file.strip()
            #replace rwkvModel. to blanck
            for k in list(w.keys()):
                if 'rwkvModel.' in k:
                    w[k.replace('rwkvModel.', '')] = w[k]
                    del w[k]
            if not args.MODEL_NAME.endswith('.pth'):
                args.MODEL_NAME += '.pth'
            import gc
            gc.collect()
            n_embd = w['emb.weight'].shape[1]
            vocab_size = w['emb.weight'].shape[0]
            dim_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            dim_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
            n_layer = 0
            keys = list(w.keys())
            version = 4
            n_head = 64
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                n_layer = max(n_layer, layer_id+1)
                if 'ln_x' in x:
                    version = max(5, version)
                if 'gate.weight' in x:
                    version = max(5.1, version)
                if int(version) == 5 and 'att.time_decay' in x:
                    n_head = w[x].shape[0]
                    if len(w[x].shape) > 1:
                        if w[x].shape[1] > 1:
                            version = max(5.2, version)
                if 'time_maa' in x:
                    version = max(6, version)
                if int(version) == 6 and 'time_faaaa' in x:
                    n_head = w[x].shape[0]

            head_size_a = dim_att // n_head
            args.n_embd = n_embd
            args.dim_att = dim_att
            args.dim_ffn = dim_ffn
            args.n_layer = n_layer
            args.version = version
            args.head_size_a = head_size_a
            args.vocab_size = vocab_size
            return w
    except Exception as e:
        traceback.print_exc()
        return None

def create_empty_args():
    import argparse
    args = argparse.Namespace()
    args.my_pos_emb = 0
    args.pre_ffn = 0
    args.head_size_divisor = 8
    args.dropout = 0
    args.head_qk = 0
    args.ctx_len = 4096
    args.grad_cp = 0
    return args

def load_ckpt_and_parse_args(ckpt_file, args):
    try:
        with torch.no_grad():
            w = torch.load(ckpt_file, map_location='cpu') # load model to CPU first
            args.MODEL_NAME = ckpt_file.strip()
            if not args.MODEL_NAME.endswith('.pth'):
                args.MODEL_NAME += '.pth'
            import gc
            gc.collect()
            n_embd = w['emb.weight'].shape[1]
            vocab_size = w['emb.weight'].shape[0]
            dim_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            dim_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
            n_layer = 0
            keys = list(w.keys())
            version = 4
            n_head = 64
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                n_layer = max(n_layer, layer_id+1)
                if 'ln_x' in x:
                    version = max(5, version)
                if 'gate.weight' in x:
                    version = max(5.1, version)
                if int(version) == 5 and 'att.time_decay' in x:
                    n_head = w[x].shape[0]
                    if len(w[x].shape) > 1:
                        if w[x].shape[1] > 1:
                            version = max(5.2, version)
                if 'time_maa' in x:
                    version = max(6, version)
                if int(version) == 6 and 'time_faaaa' in x:
                    n_head = w[x].shape[0]

            head_size_a = dim_att // n_head
            args.n_head = n_head
            args.n_embd = n_embd
            args.dim_att = dim_att
            args.dim_ffn = dim_ffn
            args.n_layer = n_layer
            args.version = version
            args.head_size_a = head_size_a
            args.vocab_size = vocab_size
            return w
    except Exception as e:
        traceback.print_exc()
        return None
    
class RwkvForSequenceClassification(pl.LightningModule):
    def __init__(self,args,num_labels,class_id = 1, pad_id = 0):
        self.args = args
        self.num_labels = num_labels
        self.pad_id = pad_id
        if not hasattr(args, 'dim_att') or args.dim_att == 0:
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn') or args.dim_ffn == 0:
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer') or args.tiny_att_layer == 0:
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim') or args.tiny_att_dim == 0:
            args.tiny_att_dim = -1
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        # Block.forward = bi_block_forward
        # from src.model import RWKV_Tmix_x060
        # RWKV_Tmix_x060.forward = bi_att_forward
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        self.drop0 = nn.Dropout(p = args.dropout)

class RwkvForClassification(pl.LightningModule):

    def __init__(self, rwkvModel, num_labels=1,class_id = 1, pad_id = 0,should_delete_head = True):
        super(RwkvForClassification, self).__init__()
        self.pad_id = pad_id
        self.class_id = class_id
        self.rwkvModel = rwkvModel
        if should_delete_head and hasattr(self.rwkvModel, 'head'):
            del self.rwkvModel.head
        self.score = nn.Linear(rwkvModel.args.n_embd, num_labels,bias=False)
        self.num_labels = num_labels
    def forward(self, idx):
        args = self.rwkvModel.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.rwkvModel.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.rwkvModel.drop0(x)
        if args.tiny_att_dim > 0:
            for block in self.rwkvModel.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for block in self.rwkvModel.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)

        x = self.rwkvModel.ln_out(x)

        #calculate the idx actual length which is first self.class_id
        idx_actual_len = torch.eq(idx, self.class_id).int().argmax(-1)
        logits = self.score(x)
        pooled_logits = logits[torch.arange(B), idx_actual_len]
        return pooled_logits
    
    def configure_optimizers(self) :
        args = self.rwkvModel.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                if ("time_mix" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_decay" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_first" in n) and (args.layerwise_lr > 0):
                    lr_3x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                    lr_decay.add(n)
                else:
                    lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        print('decay', lr_decay)
        print('1x', lr_1x)
        print('2x', lr_2x)
        print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            return DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
        else:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False)
    
    def validation_step(self, batch, batch_idx):
        idx = batch['input_ids']
        label = batch['labels']
        logits = self.forward(idx)
        if self.num_labels == 1:
            loss_fct = nn.MSELoss()
            label = label.bfloat16()
            loss = loss_fct(logits.squeeze(), label.squeeze())
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        self.log('val_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        idx = batch['input_ids']
        label = batch['labels']
        logits = self.forward(idx)
        if self.num_labels == 1:
            loss_fct = nn.MSELoss()
            label = label.bfloat16()
            loss = loss_fct(logits.squeeze(), label.squeeze())
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        self.log('train_loss', loss)
        return loss
from src.model import RWKV_Tmix_x060 as original_RWKV_Tmix_x060
class RWKV_Tmix_x060_Aggressive(original_RWKV_Tmix_x060):
    def __init__(self, args, layer_id):
        super(RWKV_Tmix_x060_Aggressive, self).__init__(args, layer_id)

    @MyFunction
    def jit_func(self, x,x1):
        B, T, C = x.size()

        xx = self.time_shift(x) - x
        xx1 = self.time_shift(x1) - x1

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xxx1 = x1 + xx1 * self.time_maa_x
        xxx1 = torch.tanh(xxx1 @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx1 = torch.bmm(xxx1, self.time_maa_w2).view(5, B, T, -1)
        mw1, mk1, mv1, mr1, mg1 = xxx1.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk1)
        xv = x + xx * (self.time_maa_v + mv1)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        xw1 = x1 + xx1 * (self.time_maa_w + mw1)
        xk1 = x1 + xx1 * (self.time_maa_k + mk1)
        xv1 = x1 + xx1 * (self.time_maa_v + mv1)
        xr1 = x1 + xx1 * (self.time_maa_r + mr1)
        xg1 = x1 + xx1 * (self.time_maa_g + mg1)

        r = self.receptance(xr)
        k = self.key(xk1)
        v = self.value(xv1)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w


    def forward(self, x,x1):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x,x1)
        from src.model import RUN_CUDA_RWKV6
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)

class OneLayerDecoder(pl.LightningModule):
    def __init__(self,args,emb):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1
        if not hasattr(args, 'emb_id'):
            args.emb_id = 1
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0
        # self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.emb = emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x060_Aggressive(args, 0)
        from src.model import RWKV_CMix_x060
        self.ffn = RWKV_CMix_x060(args, 0)
        self.drop0 = nn.Dropout(p = args.dropout)
        self.drop1 = nn.Dropout(p = args.dropout)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.ln_out = nn.LayerNorm(args.n_embd)
    
    def forward(self, x, x1):
        # print(x.shape)#B,T,C
        # print(x1.shape)#B,T
        B,T = x1.size()
        x1 = self.emb(x1)#B,T,C
        assert x1.shape == x.shape
        x = self.drop0(x + self.att(self.ln1(x), self.ln1(x1)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        x = self.ln_out(x)
        x = self.head(x)
        return x
def create_mask(x,emb_id=1,pad_id=0):
    mask = torch.ones(x.size(0),x.size(1)).to(x.device)
    mask[x == pad_id] = 0
    mask[x == emb_id] = 0
    return mask.to(torch.int)
def create_ot_mask(x,emb_id=1,mask_id=3,pad_id=0):
    mask = torch.ones(x.size(0),x.size(1)).to(x.device)
    mask[x == pad_id] = 0
    mask[x == emb_id] = 0
    mask[x == mask_id] = 0
    return mask.to(torch.int)

def reverse_x_idx(mask,max_len):
    idx_actual_len = torch.sum(mask,dim=1)
    rev_idx = []
    for i in idx_actual_len:
        reverse_idx = torch.cat([torch.arange(0,i).flip(0),torch.arange(i,max_len)],dim=0)
        rev_idx.append(reverse_idx)
    rev_idx = torch.stack(rev_idx)
    return rev_idx.to(torch.long)
def reverse_x(x,rev_idx):
    return torch.gather(x,1,rev_idx.to(x.device).unsqueeze(-1).expand(-1,-1,x.size(-1)))

def bi_att_forward(self,x,rev_idx,mask):
    B,T,C = x.size()
    H = self.n_head
    r,k,v,g,w = self.jit_func(x)
    rev_x = reverse_x(x,rev_idx)
    rev_r,rev_k,rev_v,rev_g,rev_w = self.jit_func(rev_x)
    # rev_r = reverse_x(r,rev_idx)
    # rev_k = reverse_x(k,rev_idx)
    # rev_v = reverse_x(v,rev_idx)
    # rev_w = reverse_x(w,rev_idx)
    # rev_g = reverse_x(g,rev_idx)
    from src.model import RUN_CUDA_RWKV6
    x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)
    rev_x = RUN_CUDA_RWKV6(B, T, C, H, rev_r, rev_k, rev_v, rev_w, u=self.time_faaaa)
    rev_x = reverse_x(rev_x,rev_idx)
    x = self.jit_func_2((x+rev_x)/2, g)
    return x

def bi_block_forward(self,x,rev_idx,mask):
    args = self.args
    if self.layer_id == 0:
        x = self.ln0(x)
    if args.dropout != 0:
        if self.layer_id == 0 and args.pre_ffn > 0:
            x = self.drop0(x + self.pre_ffn(self.ln1(x)))
        else:
            x = self.drop0(x+self.att(self.ln1(x),rev_idx,mask))
        x = self.drop1(x+self.ffn(self.ln2(x)))
    else:
        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.pre_ffn(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x),rev_idx,mask)
        x = x + self.ffn(self.ln2(x))
    return x

class RwkvEncoder(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att') or args.dim_att == 0:
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn') or args.dim_ffn == 0:
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer') or args.tiny_att_layer == 0:
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim') or args.tiny_att_dim == 0:
            args.tiny_att_dim = -1
        if not hasattr(args, 'emb_id'):
            args.emb_id = 1
        if not hasattr(args, 'bow_loss_weight'):
            args.bow_loss_weight = 0.1
        if not hasattr(args, 'mask_id'):
            args.mask_id = 3
        if not hasattr(args, 'share_emb'):
            args.share_emb = True
        if not hasattr(args, 'pad_id'):
            args.pad_id = 0
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        Block.forward = bi_block_forward
        from src.model import RWKV_Tmix_x060
        RWKV_Tmix_x060.forward = bi_att_forward
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        # self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.emb_id = args.emb_id

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        self.drop0 = nn.Dropout(p = args.dropout)
        if not args.share_emb:
            self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        mask = create_mask(idx,emb_id=args.emb_id,pad_id=args.pad_id)
        rev_idx = reverse_x_idx(mask,T)
        x = self.emb(idx)
        x_emb = x

        x = self.drop0(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x, rev_idx,mask)
            else:
                x = block(x,rev_idx,mask)

        x = self.ln_out(x)

        if args.head_qk > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()
            if not args.share_emb:
                x = self.head(x)+c
            else:
                x = torch.matmul(x,self.emb.weight.t())+c
        else:
            if not args.share_emb:
                x = self.head(x)
            else:   
                x = torch.matmul(x,self.emb.weight.t())
        #x is used to caclculate the MLM loss
        return x

    def training_step(self, batch, batch_idx):
        args = self.args
        encoder_input_ids,encoder_labels = batch['encoder_input_ids'],batch['encoder_labels']
        B,T = encoder_input_ids.size()
        head = self.forward(encoder_input_ids)
        enc_loss = F.cross_entropy(head.view(-1,args.vocab_size),encoder_labels.view(-1))
        return enc_loss

class RwkvEncoderBiEncoder(RwkvEncoder):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def forward(self, idx):
        x = super().forward(idx)
        actual_len = torch.eq(idx, self.args.emb_id).int().argmax(-1)
        sentence_emb = x[torch.arange(x.size(0)),actual_len]
        return sentence_emb
    
    def training_step(self, batch, batch_idx):
        query = batch["query_input_ids"]#size is (bs,seq_len_q)
        positive = batch["pos_input_ids"]#size is (bs,seq_len_p)
        negative = batch["neg_input_ids"]#size is (bs,seq_len_n)
        concatenated_inputs = torch.cat([query,positive,negative],dim=0)#size is (3*bs,seq_len)
        concatenated_embeddings = self(concatenated_inputs)#size is (3*bs,output_dim)
        total_batch_size = concatenated_embeddings.size(0)
        single_batch_size = total_batch_size // 3
        embeddings_query = concatenated_embeddings[:single_batch_size]
        embeddings_pos = concatenated_embeddings[single_batch_size:2*single_batch_size]
        embeddings_neg = concatenated_embeddings[2*single_batch_size:]


        num = len(embeddings_query)
        all_scores = None
        from torch import nn
        similarity_fct = nn.CosineSimilarity(dim=-1)
        for i in range(0, num):
            anchor_emb = embeddings_query[i].unsqueeze(0)
            pos_emb = embeddings_pos[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                one_neg_emb = embeddings_neg[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_scores is None:
                all_scores = cur_score.unsqueeze(0)
            else:
                all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

        labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
        loss = nn.CrossEntropyLoss()(all_scores, labels)

        all_another_scores = None
        for i in range(0, num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_another_scores is None:
                all_another_scores = cur_score.unsqueeze(0)
            else:
                all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)
        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)
        return loss

    
class RwkvMAEForSequenceEmbedding(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att') or args.dim_att == 0:
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn') or args.dim_ffn == 0:
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer') or args.tiny_att_layer == 0:
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim') or args.tiny_att_dim == 0:
            args.tiny_att_dim = -1
        if not hasattr(args, 'emb_id'):
            args.emb_id = 1
        if not hasattr(args, 'bi_rwkv'):
            args.bi_rwkv = False
        if not hasattr(args, 'bow_loss_weight'):
            args.bow_loss_weight = 0.1
        if not hasattr(args, 'mask_id'):
            args.mask_id = 3
        if not hasattr(args, 'pad_id'):
            args.pad_id = 0
        if not hasattr(args, 'share_emb'):
            args.share_emb = True
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        if args.bi_rwkv:
            Block.forward = bi_block_forward
            from src.model import RWKV_Tmix_x060
            RWKV_Tmix_x060.forward = bi_att_forward
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        # self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.emb_id = args.emb_id

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        self.drop0 = nn.Dropout(p = args.dropout)
        self.onelayer_decoder = OneLayerDecoder(args,self.emb)
        if not args.share_emb:
            self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def ot_embedding(self, logits, attention_mask):
        mask = (1 - attention_mask.unsqueeze(-1)) * -1000
        reps, _ = torch.max(logits + mask, dim=1)  # B V
        return reps

    def decoder_ot_loss(self, ot_embedding, bag_word_weight):
        input = F.log_softmax(ot_embedding, dim=-1)
        bow_loss = torch.mean(-torch.sum(bag_word_weight * input, dim=1))
        return bow_loss
    
    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        if args.bi_rwkv:
            mask = create_mask(idx,emb_id=args.emb_id,pad_id=args.pad_id)
            rev_idx = reverse_x_idx(mask,T)
        x = self.emb(idx)
        x_emb = x

        x = self.drop0(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                if args.bi_rwkv:
                    x = deepspeed.checkpointing.checkpoint(block, x, rev_idx,mask)
                else:
                    x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                if args.bi_rwkv:
                    x = block(x,rev_idx,mask)
                else:
                    x = block(x)

        x = self.ln_out(x)
        #actual len is the first self.emb_id
        idx_actual_len = torch.eq(idx, self.emb_id).int().argmax(-1)
        #get the last token embedding
        seq_emb = x[torch.arange(x.size(0)), idx_actual_len]
        #add the last token embeddings to the x[:,:idx_actual_len-1]
        if not args.bi_rwkv:
            x = x + seq_emb.unsqueeze(1).expand(-1,T,-1)

        if args.head_qk > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

            if not args.share_emb:
                x = self.head(x)+c
            else:
                x = torch.matmul(x,self.emb.weight.t())+c
        else:
            if not args.share_emb:
                x = self.head(x)
            else:   
                x = torch.matmul(x,self.emb.weight.t())
        #x is used to caclculate the MLM loss
        return seq_emb, x,mask

    def training_step(self, batch, batch_idx):
        args = self.args
        encoder_input_ids,encoder_labels,decoder_input_ids,decoder_labels = batch['encoder_input_ids'],batch['encoder_labels'],batch['decoder_input_ids'],batch['decoder_labels']
        B,T = encoder_input_ids.size()
        h,head,mask = self.forward(encoder_input_ids)
        enc_loss = F.cross_entropy(head.view(-1,args.vocab_size),encoder_labels.view(-1))
        del encoder_labels
        torch.cuda.empty_cache()
        h = h.unsqueeze(1).expand(-1,T,-1)
        decoder_out = self.onelayer_decoder(h,decoder_input_ids)
        decoder_loss = F.cross_entropy(decoder_out.view(-1,args.vocab_size),decoder_labels.view(-1))
        del decoder_input_ids, decoder_labels,decoder_out
        torch.cuda.empty_cache()
        returned_loss = {}
        if args.dup_mae:
            # ot_mask = create_ot_mask(encoder_input_ids,emb_id=args.emb_id,mask_id=args.mask_id)
            ot_mask = mask
            ot_embedding = self.ot_embedding(head, ot_mask)
            bag_word_weight = batch['bag_word_weight']
            bow_loss = self.decoder_ot_loss(ot_embedding, bag_word_weight)
            del bag_word_weight,encoder_input_ids
            del ot_embedding
            torch.cuda.empty_cache()
            returned_loss['bow_loss'] = bow_loss*args.bow_loss_weight
            loss = enc_loss + decoder_loss + bow_loss
        else:
            loss = enc_loss + decoder_loss
        returned_loss['enc_loss'] = enc_loss
        returned_loss['decoder_loss'] = decoder_loss
        returned_loss['loss'] = loss
        self.log('train_loss', loss)
        return returned_loss
    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
    
class RwkvInstructorForSequenceEmbedding(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1
        if not hasattr(args, 'emb_id'):
            args.emb_id = 1
        if not hasattr(args, 'output_dim'):
            args.output_dim = 1024
        if not hasattr(args, 'pooling_type'):
            args.pooling_type = 'avg'
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0
        H =  args.dim_att // args.head_size_a
        # RWKV_Tmix_x060_infctx.forward = att_masked_forward
        if args.bi_rwkv:
            Block.forward = bi_block_forward
            from src.model import RWKV_Tmix_x060
            RWKV_Tmix_x060.forward = bi_att_forward
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.dense = nn.Linear(args.n_embd, args.output_dim,bias=False)
        # self.dense = nn.Linear(H*args.head_size_a*args.head_size_a, args.output_dim,bias=False)
        self.ln_dense = nn.LayerNorm(args.output_dim)
        self.output_dim = args.output_dim
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        # self.states_cnn = StatesCNN(args.n_layer,H,args.head_size_a,args.output_dim)
    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            print(f'init {n}')
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
    
    def pooling(self, x,actual_len):
        args = self.args
        if args.pooling_type == 'weightedmean':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the actual length of each sequence
            #weights is (bs,seq_len) float tensor which indicates the weight of each token, the weight[i] = (i+1)/actual_len[i], the last token embedding is 1 and others are degraded by the distance to the last token 
            #create a mask to mask the padding token
            mask = torch.arange(x.size(1),device = x.device) <= actual_len.unsqueeze(1)
            weights = torch.arange(1,x.size(1)+1,device = x.device).unsqueeze(0).float() / actual_len.unsqueeze(1).float()
            #mask weights to zero according mask
            weights = weights * mask.float()
            #add the sum of token embeddings from 0 to actual len as the final embedding 
            x = torch.sum(x * weights.unsqueeze(-1),dim=1)
            x = x / actual_len.unsqueeze(1).float()
            return x.bfloat16()
        elif args.pooling_type == 'lasttoken':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the index of last token
            #return the last token embedding
            x = x[torch.arange(x.size(0)),actual_len]
            return x
        elif args.pooling_type == 'avg':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the actual length of each sequence
            #return the average of all token embeddings
            #mask is [[1,1,1,...,0,0,0],[1,1,1,...,0,0,0],...,[1,1,1,...,0,0,0]] which is used to mask the padding token
            mask = torch.ones((x.size(0),x.size(1)),device = x.device)
            col_indices = torch.arange(mask.size(1)).unsqueeze(0).to(mask.device)
            mask_indices = col_indices >= actual_len.unsqueeze(1)
            mask[mask_indices] = 0
            x = torch.sum(x*mask.unsqueeze(-1),dim=1) / actual_len.unsqueeze(1).float()
            return x.bfloat16()
    
    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        if args.bi_rwkv:
            mask = create_mask(idx,emb_id=args.emb_id)
            rev_idx = reverse_x_idx(mask,T)
        x = self.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.drop0(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                if args.bi_rwkv:
                    x = deepspeed.checkpointing.checkpoint(block, x, rev_idx,mask)
                else:
                    x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                if args.bi_rwkv:
                    x = block(x,rev_idx,mask)
                else:
                    x = block(x)

            # if args.grad_cp == 1:
            #     x = deepspeed.checkpointing.checkpoint(block, x)
            # else:
            #     x = block(x)

        x = self.ln_out(x)

        #calculate the idx actual length which is first self.embedding_id
        idx_actual_len = torch.eq(idx, args.emb_id).int().argmax(-1)
        x = self.pooling(x,idx_actual_len)
        x = self.dense(x)
        x = self.ln_dense(x)
        return x

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def configure_optimizers(self) :
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                if ("time_mix" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_decay" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_first" in n) and (args.layerwise_lr > 0):
                    lr_3x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                    lr_decay.add(n)
                else:
                    lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]
        # print('optim_groups', optim_groups)
        # print(lr_1x)
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if torch.backends.mps.is_available():
                from torch.optim import AdamW,Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=args.weight_decay)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
            else:
                if self.deepspeed_offload:
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    return DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
                return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if torch.backends.mps.is_available():
                from torch.optim import AdamW, Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=0)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,  weight_decay=0, amsgrad=False)
            else:
                if self.deepspeed_offload:
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False)
                return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, adam_w_mode=False, weight_decay=0, amsgrad=False)
    def training_step(self, batch, batch_idx):
        query = batch["query_input_ids"]#size is (bs,seq_len_q)
        positive = batch["pos_input_ids"]#size is (bs,seq_len_p)
        negative = batch["neg_input_ids"]#size is (bs,seq_len_n)
        concatenated_inputs = torch.cat([query,positive,negative],dim=0)#size is (3*bs,seq_len)
        concatenated_embeddings = self(concatenated_inputs)#size is (3*bs,output_dim)
        total_batch_size = concatenated_embeddings.size(0)
        single_batch_size = total_batch_size // 3
        embeddings_query = concatenated_embeddings[:single_batch_size]
        embeddings_pos = concatenated_embeddings[single_batch_size:2*single_batch_size]
        embeddings_neg = concatenated_embeddings[2*single_batch_size:]
        # embeddings_query= self(query)#size is (bs,output_dim)
        # torch.cuda.empty_cache()
        # embeddings_pos = self(positive)#size is (bs,output_dim)
        # torch.cuda.empty_cache()
        # embeddings_neg = self(negative)#size is (bs,output_dim)
        # torch.cuda.empty_cache()

        num = len(embeddings_query)
        all_scores = None
        from torch import nn
        similarity_fct = nn.CosineSimilarity(dim=-1)
        for i in range(0, num):
            anchor_emb = embeddings_query[i].unsqueeze(0)
            pos_emb = embeddings_pos[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                one_neg_emb = embeddings_neg[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_scores is None:
                all_scores = cur_score.unsqueeze(0)
            else:
                all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

        labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
        loss = nn.CrossEntropyLoss()(all_scores, labels)

        all_another_scores = None
        for i in range(0, num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_another_scores is None:
                all_another_scores = cur_score.unsqueeze(0)
            else:
                all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)
        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)
        return loss

class RwkvStatesForSequenceEmbedding(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1
        if not hasattr(args, 'emb_id'):
            args.emb_id = 1
        if not hasattr(args, 'output_dim'):
            args.output_dim = 1024
        if not hasattr(args, 'pooling_type'):
            args.pooling_type = 'avg'
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0
        H =  args.dim_att // args.head_size_a
        # RWKV_Tmix_x060_infctx.forward = att_masked_forward
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.dense = nn.Linear(args.n_embd, args.output_dim,bias=False)
        # self.dense = nn.Linear(H*args.head_size_a*args.head_size_a, args.output_dim,bias=False)
        self.ln_dense = nn.LayerNorm(args.output_dim)
        self.output_dim = args.output_dim
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        # self.states_cnn = StatesCNN(args.n_layer,H,args.head_size_a,args.output_dim)
    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            print(f'init {n}')
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
    
    def pooling(self, x,actual_len):
        args = self.args
        if args.pooling_type == 'weightedmean':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the actual length of each sequence
            #weights is (bs,seq_len) float tensor which indicates the weight of each token, the weight[i] = (i+1)/actual_len[i], the last token embedding is 1 and others are degraded by the distance to the last token 
            #create a mask to mask the padding token
            mask = torch.arange(x.size(1),device = x.device) <= actual_len.unsqueeze(1)
            weights = torch.arange(1,x.size(1)+1,device = x.device).unsqueeze(0).float() / actual_len.unsqueeze(1).float()
            #mask weights to zero according mask
            weights = weights * mask.float()
            #add the sum of token embeddings from 0 to actual len as the final embedding 
            x = torch.sum(x * weights.unsqueeze(-1),dim=1)
            x = x / actual_len.unsqueeze(1).float()
            return x.bfloat16()
        elif args.pooling_type == 'lasttoken':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the index of last token
            #return the last token embedding
            x = x[torch.arange(x.size(0)),actual_len]
            return x
        elif args.pooling_type == 'avg':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the actual length of each sequence
            #return the average of all token embeddings
            #mask is [[1,1,1,...,0,0,0],[1,1,1,...,0,0,0],...,[1,1,1,...,0,0,0]] which is used to mask the padding token
            mask = torch.ones((x.size(0),x.size(1)),device = x.device)
            col_indices = torch.arange(mask.size(1)).unsqueeze(0).to(mask.device)
            mask_indices = col_indices >= actual_len.unsqueeze(1)
            mask[mask_indices] = 0
            x = torch.sum(x*mask.unsqueeze(-1),dim=1) / actual_len.unsqueeze(1).float()
            return x.bfloat16()
    
    def forward(self, idx):
        args = self.args
        T_train = args.chunk_ctx 
        B, T = idx.shape
        C = args.n_embd
        H =  args.dim_att // args.head_size_a
        assert C==H*args.head_size_a
        states = BlockStateList.create(args.n_layer, B, C, H, idx.device,
                self.emb.weight.dtype)
        token_amount = 0
        i = 0
        x_final = torch.zeros((B,T,args.n_embd),device = idx.device,dtype = torch.bfloat16)
        for i in range(math.ceil(T / T_train)):
            idx_chunk = idx[:, i * T_train:(i + 1) * T_train]
            x_chunk,new_shift_states,new_wkv_states = self.forward_innner(idx_chunk,states.shift_states,states.wkv_states)
            states = BlockStateList(new_shift_states, new_wkv_states)
            x_final[:,i * T_train:(i + 1) * T_train,:] = x_chunk
        idx_actual_len = torch.eq(idx, args.emb_id).int().argmax(-1)
        seq_emb = self.pooling(x_final,idx_actual_len)
        x = self.dense(seq_emb)
        x = self.ln_dense(x)
        return x,states.shift_states,states.wkv_states
    def forward_innner(self, idx, last_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor):
        args = self.args
        B, T = idx.size()
        assert T <= args.chunk_ctx, "Cannot forward, model ctx_len is exhausted."
        C = args.n_embd
        H =  args.dim_att // args.head_size_a
        assert C==H*args.head_size_a
        
        x = self.emb(idx)
        x_emb = x
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                        x.device, x.dtype)
        if args.dropout > 0:
            x = self.drop0(x)

        for i, (block, block_state) in enumerate(zip(self.blocks,
            BlockStateList(last_shift_states, last_wkv_states))):
            # x = x.to(block.device)
            if args.grad_cp == 1 and i > 0:  # and i < len(self.blocks)-1
                x, new_block_state = torch_checkpoint(block, x, block_state, use_reentrant=False)
            else:
                x, new_block_state = block(x, block_state)
            new_states[i] = new_block_state

        x = self.ln_out(x)


        return x, new_states.shift_states, new_states.wkv_states

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def configure_optimizers(self) :
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                if ("time_mix" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_decay" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_first" in n) and (args.layerwise_lr > 0):
                    lr_3x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                    lr_decay.add(n)
                else:
                    lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]
        # print('optim_groups', optim_groups)
        # print(lr_1x)
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if torch.backends.mps.is_available():
                from torch.optim import AdamW,Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=args.weight_decay)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
            else:
                if self.deepspeed_offload:
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    return DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
                return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if torch.backends.mps.is_available():
                from torch.optim import AdamW, Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=0)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,  weight_decay=0, amsgrad=False)
            else:
                if self.deepspeed_offload:
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False)
                return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, adam_w_mode=False, weight_decay=0, amsgrad=False)
    def training_step(self, batch, batch_idx):
        query = batch["query_input_ids"]#size is (bs,seq_len_q)
        positive = batch["pos_input_ids"]#size is (bs,seq_len_p)
        negative = batch["neg_input_ids"]#size is (bs,seq_len_n)
        embeddings_query,_,_ = self(query)#size is (bs,output_dim)
        embeddings_pos,_,_ = self(positive)#size is (bs,output_dim)
        embeddings_neg,_,_ = self(negative)#size is (bs,output_dim)

        num = len(embeddings_query)
        all_scores = None
        from torch import nn
        similarity_fct = nn.CosineSimilarity(dim=-1)
        for i in range(0, num):
            anchor_emb = embeddings_query[i].unsqueeze(0)
            pos_emb = embeddings_pos[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                one_neg_emb = embeddings_neg[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_scores is None:
                all_scores = cur_score.unsqueeze(0)
            else:
                all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

        labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
        loss = nn.CrossEntropyLoss()(all_scores, labels)

        all_another_scores = None
        for i in range(0, num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_another_scores is None:
                all_another_scores = cur_score.unsqueeze(0)
            else:
                all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)
        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)
        return loss
    
class RwkvForSequenceEmbedding(pl.LightningModule):

    def __init__(self, rwkvModel,embedding_id = 1, pad_id = 0,should_delete_head = True,pooling_type='weightedmean',add_mlp = False,is_in_batch_negative = False,output_dim = 0):
        super(RwkvForSequenceEmbedding, self).__init__()
        self.pad_id = pad_id
        self.rwkvModel = rwkvModel
        self.embedding_id = embedding_id
        self.pooling_type = pooling_type
        self.add_mlp = add_mlp
        self.is_in_batch_negative = is_in_batch_negative
        if add_mlp:
            if output_dim == 0:
                output_dim = rwkvModel.args.n_embd
            self.dense = nn.Linear(rwkvModel.args.n_embd, output_dim)
            self.activation = nn.Tanh()
        if should_delete_head and hasattr(self.rwkvModel, 'head'):
            del self.rwkvModel.head

    def pooling(self, x,actual_len):
        if self.pooling_type == 'weightedmean':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the actual length of each sequence
            #weights is (bs,seq_len) float tensor which indicates the weight of each token, the weight[i] = (i+1)/actual_len[i], the last token embedding is 1 and others are degraded by the distance to the last token 
            #create a mask to mask the padding token
            mask = torch.arange(x.size(1),device = x.device) <= actual_len.unsqueeze(1)
            weights = torch.arange(1,x.size(1)+1,device = x.device).unsqueeze(0).float() / actual_len.unsqueeze(1).float()
            #mask weights to zero according mask
            weights = weights * mask.float()
            #add the sum of token embeddings from 0 to actual len as the final embedding 
            x = torch.sum(x * weights.unsqueeze(-1),dim=1)
            x = x / actual_len.unsqueeze(1).float()
            return x.bfloat16()
        elif self.pooling_type == 'lasttoken':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the index of last token
            #return the last token embedding
            x = x[torch.arange(x.size(0)),actual_len]
            return x
        elif self.pooling_type == 'avg':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the actual length of each sequence
            #return the average of all token embeddings
            #mask is [[1,1,1,...,0,0,0],[1,1,1,...,0,0,0],...,[1,1,1,...,0,0,0]] which is used to mask the padding token
            mask = torch.ones((x.size(0),x.size(1)),device = x.device)
            col_indices = torch.arange(mask.size(1)).unsqueeze(0).to(mask.device)
            mask_indices = col_indices >= actual_len.unsqueeze(1)
            mask[mask_indices] = 0
            x = torch.sum(x*mask.unsqueeze(-1),dim=1) / actual_len.unsqueeze(1).float()
            return x.bfloat16()
    def forward(self, idx):
        args = self.rwkvModel.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.rwkvModel.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.rwkvModel.drop0(x)
        if args.tiny_att_dim > 0:
            for block in self.rwkvModel.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for block in self.rwkvModel.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)

        x = self.rwkvModel.ln_out(x)

        #calculate the idx actual length which is first self.embedding_id
        idx_actual_len = torch.eq(idx, self.embedding_id).int().argmax(-1)
        x = self.pooling(x,idx_actual_len)
        if self.add_mlp:
            x = self.activation(self.dense(x))
        return x
    
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def configure_optimizers(self) :
        args = self.rwkvModel.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                if ("time_mix" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_decay" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_first" in n) and (args.layerwise_lr > 0):
                    lr_3x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                    lr_decay.add(n)
                else:
                    lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]
        # print('optim_groups', optim_groups)
        # print(lr_1x)
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if torch.backends.mps.is_available():
                from torch.optim import AdamW,Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=args.weight_decay)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
            else:
                if self.deepspeed_offload:
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    return DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
                return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if torch.backends.mps.is_available():
                from torch.optim import AdamW, Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=0)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,  weight_decay=0, amsgrad=False)
            else:
                if self.deepspeed_offload:
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False)
                return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, adam_w_mode=False, weight_decay=0, amsgrad=False)
    
    def validation_step(self, batch, batch_idx):
        query = batch["query"]#size is (bs,seq_len)
        positive = batch["positive"]#size is (bs,seq_len)
        if "negative" in batch:
            negative = batch["negative"]
        else:
            negative = None
        query_embeddings = self.forward(query)#size is (bs,emb_dim)
        positive_embeddings = self.forward(positive)#size is (bs,emb_dim)
        from sentence_transformers import util
        similarity_fct=util.cos_sim
        scores = similarity_fct(query_embeddings, positive_embeddings)*20
        if negative is not None:
            pairwise_fct = util.pairwise_cos_sim
            negative_embeddings = self.forward(negative)
            negative_scores = pairwise_fct(query_embeddings, negative_embeddings).unsqueeze(1)*20#score is (bs,1)
            scores = torch.cat([scores,negative_scores],dim=1)#scores size is (bs,bs+1)
        labels = torch.arange(0, scores.shape[0], dtype=torch.long).to(scores.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(scores, labels)
        self.log("val_loss", loss,sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        query = batch["query"]#size is (bs,seq_len)
        positive = batch["positive"]#size is (bs,seq_len)
        if "negative" in batch:
            negative = batch["negative"]
        else:
            negative = None
        if negative is not None:
            embeddings = self.forward(torch.cat([query, positive, negative], dim=0))  # size is (3*bs, emb_dim)
            query_embeddings = embeddings[:batch["query"].size(0)]
            positive_embeddings = embeddings[batch["query"].size(0):batch["query"].size(0)+batch["positive"].size(0)]
            negative_embeddings = embeddings[batch["query"].size(0)+batch["positive"].size(0):]
        else:
            embeddings = self.forward(torch.cat([query, positive], dim=0))  # size is (2*bs, emb_dim)
            query_embeddings = embeddings[:batch["query"].size(0)]
            positive_embeddings = embeddings[batch["query"].size(0):]
            negative_embeddings = None
        if self.is_in_batch_negative:
            from sentence_transformers import util
            similarity_fct=util.cos_sim
            scores = similarity_fct(query_embeddings, positive_embeddings)*20
            if negative is not None:
                pairwise_fct = util.pairwise_cos_sim
                negative_scores = pairwise_fct(query_embeddings, negative_embeddings).unsqueeze(1)*20#score is (bs,1)
                scores = torch.cat([scores,negative_scores],dim=1)#scores size is (bs,bs+1)
            labels = torch.arange(0, scores.shape[0], dtype=torch.long).to(scores.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(scores, labels)
            self.log("train_loss", loss)
            return loss
        else:
        
        # It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        # the expected similarity score between the pair.

        # It computes the following loss function:

        # ``loss = logsum(1+exp(s(k,l)-s(i,j))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        # batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        # pairs of input pairs in the batch that match this condition.

        # Anecdotal experiments show that this loss function produces a more powerful training signal than :class:`CosineSimilarityLoss`,
        # resulting in faster convergence and a final model with superior performance. Consequently, CoSENTLoss may be used
        # as a drop-in replacement for :class:`CosineSimilarityLoss` in any training script.

        # :param model: SentenceTransformerModel
        # :param similarity_fct: Function to compute the PAIRWISE similarity between embeddings. Default is ``util.pairwise_cos_sim``.
        # :param scale: Output of similarity function is multiplied by scale value. Represents the inverse temperature.

        # References:
        #     - For further details, see: https://kexue.fm/archives/8847

        # Requirements:
        #     - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

  
            labels = torch.ones(positive_embeddings.shape[0]).to(positive_embeddings.device)
            from sentence_transformers import util
            similarity_fct = util.pairwise_cos_sim
            scores = similarity_fct(query_embeddings, positive_embeddings)
            if negative is not None:
                scores = torch.cat([scores,similarity_fct(query_embeddings, negative_embeddings)])
                labels = torch.cat([labels,torch.full((negative_embeddings.shape[0],),-1).to(positive_embeddings.device)])
            
            scores = scores * 20
            scores = scores[:, None] - scores[None, :]

            # label matrix indicating which pairs are relevant
            labels = labels[:, None] < labels[None, :]
            labels = labels.bfloat16()

            # mask out irrelevant pairs so they are negligible after exp()
            scores = scores - (1 - labels) * 1e12

            # append a zero as e^0 = 1
            scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
            loss = torch.logsumexp(scores, dim=0)/scores.shape[0]
            self.log("train_loss", loss)
            return loss