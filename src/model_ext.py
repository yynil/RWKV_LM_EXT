import torch
import torch.nn as nn

import deepspeed

import os
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam
from sentence_transformers.util import pairwise_dot_score
import traceback

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
        print('optim_groups', optim_groups)
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            return DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
        else:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False)
    
    def training_step(self, batch, batch_idx):
        idx, label = batch
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


class RwkvForSequenceEmbedding(pl.LightningModule):

    def __init__(self, rwkvModel,embedding_id = 1, pad_id = 0,should_delete_head = True,pooling_type='weightedmean',add_mlp = False,is_in_batch_negative = False):
        super(RwkvForSequenceEmbedding, self).__init__()
        self.pad_id = pad_id
        self.rwkvModel = rwkvModel
        self.embedding_id = embedding_id
        self.pooling_type = pooling_type
        self.add_mlp = add_mlp
        self.is_in_batch_negative = is_in_batch_negative
        if add_mlp:
            self.dense = nn.Linear(rwkvModel.args.n_embd, rwkvModel.args.n_embd)
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
            return x
        elif self.pooling_type == 'lasttoken':
            #x is (bs,seq_len,emb_dim)
            #actual_len is (bs,) int tensor which indicates the index of last token
            #return the last token embedding
            x = x[torch.arange(x.size(0)),actual_len]
            return x
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
            x = self.activation(self.dense(x.float())).bfloat16()
        return x
    
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
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if torch.backends.mps.is_available():
                from torch.optim import AdamW,Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=args.weight_decay)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
            else:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                return DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
                # return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if torch.backends.mps.is_available():
                from torch.optim import AdamW, Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=0)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,  weight_decay=0, amsgrad=False)
            else:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False)
                # return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, adam_w_mode=False, weight_decay=0, amsgrad=False)
    
    def training_step(self, batch, batch_idx):
        query = batch["query"]#size is (bs,seq_len)
        positive = batch["positive"]#size is (bs,seq_len)
        if "negative" in batch:
            negative = batch["negative"]
        else:
            negative = None
        query_embeddings = self.forward(query)#size is (bs,emb_dim)
        positive_embeddings = self.forward(positive)#size is (bs,emb_dim)
        if self.is_in_batch_negative:
            if negative is not None:
                negative_embeddings = self.forward(negative)#size is (bs,emb_dim)
                positive_embeddings = torch.cat([positive_embeddings,negative_embeddings])#size is (2*bs,emb_dim)
            from sentence_transformers import util
            similarity_fct=util.cos_sim
            scores = similarity_fct(query_embeddings, positive_embeddings)*20
            labels = torch.arange(0, scores.shape[0], dtype=torch.long).to(scores.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(scores, labels)
            self.log("train_loss", loss)
            return loss
        else:
            labels = torch.zeros(positive_embeddings.shape[0]).to(positive_embeddings.device)
            from sentence_transformers import util
            similarity_fct = util.pairwise_cos_sim
            scores = similarity_fct(query_embeddings, positive_embeddings)
            if negative is not None:
                negative_embeddings = self.forward(negative)
                scores = torch.cat([scores,similarity_fct(query_embeddings, negative_embeddings)])
                labels = torch.cat([labels,torch.ones(negative_embeddings.shape[0]).to(positive_embeddings.device)])
            labels = labels.bfloat16()
            
            scores = scores * 20
            scores = scores[:, None] - scores[None, :]

            # label matrix indicating which pairs are relevant
            labels = labels[:, None] < labels[None, :]
            labels = labels.bfloat16()

            # mask out irrelevant pairs so they are negligible after exp()
            scores = scores - (1 - labels) * 1e12

            # append a zero as e^0 = 1
            scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
            loss = torch.logsumexp(scores, dim=0)

            return loss