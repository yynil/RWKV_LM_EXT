import torch
import torch.nn as nn
from torch.nn import functional as F
import os

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
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

if 'NO_CUDA'in os.environ and os.environ['NO_CUDA'] == '1':
    def run_rwkv6_forward(r,k,v,w,u):
        B, T, C = r.shape
        H = C // HEAD_SIZE
        
        # 重塑张量
        r = r.view(B, T, H, HEAD_SIZE)
        k = k.view(B, T, H, HEAD_SIZE)
        v = v.view(B, T, H, HEAD_SIZE)
        w = w.view(B, T, H, HEAD_SIZE)

        # 处理w，匹配CUDA实现
        ew = torch.exp(-torch.exp(w))

        y = torch.zeros((B, T, H, HEAD_SIZE), device=r.device, dtype=torch.float32)
        
        # 初始化状态张量，包含所有batch和head
        state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, dtype=torch.float32, device=r.device)

        for t in range(T):
            rt = r[:, t]  # Shape: [B, H, HEAD_SIZE]
            kt = k[:, t]  # Shape: [B, H, HEAD_SIZE]
            vt = v[:, t]  # Shape: [B, H, HEAD_SIZE]
            wt = ew[:, t]  # Shape: [B, H, HEAD_SIZE]
            
            for j in range(HEAD_SIZE):
                v_h = vt[:, :, j].unsqueeze(2)  # Shape: [B, H, 1]
                x = kt * v_h  # Shape: [B, H, HEAD_SIZE]
                y_local = torch.sum(rt * (x * u + state[:, :, j]), dim=2)  # Shape: [B, H]
                state[:, :, j] = state[:, :, j] * wt + x
                y[:, t, :, j] = y_local

        return y.view(B, T, C)
        
    def bi_att_forward_batch(self,x,rev_idx,mask):
        B,T,C = x.size()
        H = self.n_head
        r,k,v,g,w = self.jit_func(x)
        rev_x = reverse_x(x,rev_idx)
        rev_r,rev_k,rev_v,rev_g,rev_w = self.jit_func(rev_x)
        u = self.time_faaaa
        x = run_rwkv6_forward(r, k, v, w, u)
        rev_x = run_rwkv6_forward(rev_r, rev_k, rev_v, rev_w, u)
        rev_x = reverse_x(rev_x,rev_idx)
        x = self.jit_func_2((x+rev_x)/2, g)
        return x
else:
    def bi_att_forward_batch(self,x,rev_idx,mask):
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


class BiRWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
            if args.n_embd==4096:
                TIME_MIX_EXTRA_DIM = TIME_MIX_EXTRA_DIM*2
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, args.n_embd).uniform_(-1e-4, 1e-4))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            TIME_DECAY_EXTRA_DIM = 64
            if args.n_embd==4096:
                TIME_DECAY_EXTRA_DIM = TIME_DECAY_EXTRA_DIM*2
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, args.dim_att).uniform_(-1e-4, 1e-4))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x,rev_idx,mask):
        return bi_att_forward_batch(self,x,rev_idx,mask)


class BiRWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class BiBlock(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        self.att = BiRWKV_Tmix_x060(args, layer_id)

        self.ffn = BiRWKV_CMix_x060(args, layer_id)

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))


    def forward(self,x,rev_idx,mask):
        args = self.args
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.pre_ffn(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x),rev_idx,mask)
        x = x + self.ffn(self.ln2(x))
        return x


class RwkvEncoder(nn.Module):
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
        # from src.model import RWKV_Tmix_x060,Block
        # Block.forward = bi_block_forward_batch
        # RWKV_Tmix_x060.forward = bi_att_forward_batch
        self.blocks = nn.ModuleList([BiBlock(args, i) for i in range(args.n_layer)])
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
    def encode_sentence(self, idx):
        _,embs = self.forward(idx,True)
        #get the position of emb_id
        position = torch.eq(idx, self.emb_id).int().argmax(-1)
        return embs[torch.arange(embs.size(0)), position]
    def forward(self, idx,return_logits=False):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        mask = create_mask(idx,emb_id=args.emb_id,pad_id=args.pad_id)
        rev_idx = reverse_x_idx(mask,T)
        x = self.emb(idx)
        x_emb = x

        x = self.drop0(x)
        for block in self.blocks:
            x = block(x,rev_idx,mask)

        x = self.ln_out(x)
        logits = x
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
        if return_logits:
            return x,logits
        return x