########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
from typing import Dict, List, Optional, Union
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(parent_path)
print(f'add path: {parent_path} to sys.path')
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from peft.tuners.lora.layer import LoraLayer
def __nop(ob):
    return ob
from torch._lowrank import svd_lowrank

MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

if 'x060' in os.environ["RWKV_MY_TESTING"]:
    rwkv6 = load(name="rwkv6", sources=[f"{parent_path}/cuda/rwkv6_op.cpp", f"{parent_path}/cuda/rwkv6.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])
    print(f"Loaded RWKV6 CUDA Kernel:{rwkv6}")
    class RWKV_6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, state, r, k, v, w, u):
            with torch.no_grad():
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert state.dtype == torch.float32
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                eew = torch.exp(-torch.exp(w.float())).contiguous()

                y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
                if r.dtype == torch.bfloat16:
                    rwkv6.forward_bf16(B, T, C, H, state, r, k, v, eew, u, y)
                elif r.dtype == torch.float16:
                    rwkv6.forward_fp16(B, T, C, H, state, r, k, v, eew, u, y)
                elif r.dtype == torch.float32:
                    rwkv6.forward_fp32(B, T, C, H, state, r, k, v, eew, u, y)
                return y,state

    def RUN_RWKV_6(B, T, C, H, state, r, k, v, w, u):
        return RWKV_6.apply(B, T, C, H, state, r, k, v, w, u)
else:
    wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
        
    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.bfloat16
                assert k.dtype == torch.bfloat16
                assert v.dtype == torch.bfloat16
                assert w.dtype == torch.bfloat16
                assert u.dtype == torch.bfloat16
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ew = (-torch.exp(w.float())).contiguous()
                eew = (torch.exp(ew)).contiguous()
                ctx.save_for_backward(r, k, v, eew, ew, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.bfloat16
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, eew, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
        return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_TimeMix_RWKV5(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        assert HEAD_SIZE == self.head_size # change HEAD_SIZE to match args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

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
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa)

        return self.jit_func_2(x, g)

class RWKV_Tmix_x060(MyModule):
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

    @MyFunction
    def jit_func(self, x,state_xx):
        T, C = x.size()

        xx = torch.cat((state_xx.unsqueeze(0),x[:-1,:])) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(T,5,-1).transpose(0,1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, T, -1)
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

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x,state_xx,state_kv):
        T, C = x.size()
        H = self.n_head
        xx = x[-1,:]
        r, k, v, g, w = self.jit_func(x,state_xx)
        x,s = RUN_RWKV_6(1, T, C, H, state_kv.transpose(-1,-2).contiguous(),r, k, v, w, self.time_faaaa)
        s = s.transpose(-1,-2)
        x = self.jit_func_2(x, g).squeeze(0)
        return x,xx,s

########################################################################################################

class RWKV_ChannelMix(MyModule):
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
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class RWKV_CMix_x060(MyModule):
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

    @MyFunction
    def forward(self, x,state_ffn):
        xx = torch.cat((state_ffn.unsqueeze(0),x[:-1,:])) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return (torch.sigmoid(self.receptance(xr)) * kv).squeeze(0),x[-1,:]

########################################################################################################

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
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

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_Tmix_x060(args, layer_id)
            else:
                self.att = RWKV_TimeMix_RWKV5(args, layer_id)

        if 'g' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        else:
            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                self.ffn = RWKV_CMix_x060(args, layer_id)
            else:
                self.ffn = RWKV_ChannelMix(args, layer_id)
        
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x,state, x_emb=None):
        args = self.args
        T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            x_,x_x,state_kv = self.att(self.ln1(x),state[0],state[1])
            x = x + x_
        x_,state_ffn = self.ffn(self.ln2(x),state[2])
        x = x + x_

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x,x_x,state_kv,state_ffn


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

def create_mask(x,emb_id=1):
    mask = torch.ones(x.size(0),x.size(1)).to(x.device)
    mask[x == 0] = 0
    mask[x == emb_id] = 0
    return mask.to(torch.int)
def create_ot_mask(x,emb_id=1,mask_id=3):
    mask = torch.ones(x.size(0),x.size(1)).to(x.device)
    mask[x == 0] = 0
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

def bi_block_forward_batch(self,x,rev_idx,mask):
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
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        from src.model import RWKV_Tmix_x060,Block
        Block.forward = bi_block_forward_batch
        RWKV_Tmix_x060.forward = bi_att_forward_batch
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.emb_id = args.emb_id

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        self.drop0 = nn.Dropout(p = args.dropout)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        mask = create_mask(idx,emb_id=args.emb_id)
        rev_idx = reverse_x_idx(mask,T)
        x = self.emb(idx)
        x_emb = x

        x = self.drop0(x)
        for block in self.blocks:
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

            x = self.head(x) + c
        else:
            x = self.head(x)
        #x is used to caclculate the MLM loss
        return x
    
class RWKV(torch.nn.Module):
    def __init__(self, args):
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
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)


    def forward(self, idx,state=None):
        with torch.no_grad():
            args = self.args
            T = idx.size()[0]
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
            if state is None:
                state = [None] * args.n_layer * 3
                for i in range(args.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
                    state[i*3+0] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*3+1] = torch.zeros((args.n_head, args.n_att//args.n_head, args.n_att//args.n_head), dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*3+2] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
            x = self.emb(idx)
            x_emb = x
            layer_id = 0
            if args.tiny_att_dim > 0:
                for block in self.blocks:
                    x,x_x,state_kv,state_ffn = block(x,state[layer_id*3:layer_id*3+3], x_emb)
                    state[layer_id*3+0] = x_x
                    state[layer_id*3+1] = state_kv
                    state[layer_id*3+2] = state_ffn
                    layer_id += 1
            else:
                for block in self.blocks:
                    x,x_x,state_kv,state_ffn = block(x,state[layer_id*3:layer_id*3+3])
                    state[layer_id*3+0] = x_x
                    state[layer_id*3+1] = state_kv
                    state[layer_id*3+2] = state_ffn
                    layer_id += 1

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

                x = self.head(x) + c
            else:
                x = self.head(x)

            return x[-1,:],state
        
def bi_block_forward(self, x,state, x_emb=None):
    args = self.args
    T, C = x.size()
    if self.layer_id == 0:
        x = self.ln0(x)
        if args.my_pos_emb > 0:
            pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
            x = x + pos_emb

    if self.layer_id == 0 and args.pre_ffn > 0:
        x = x + self.ffnPre(self.ln1(x))
    else:
        x_,x_x,state_kv,x_x_rev,state_kv_rev = self.att(self.ln1(x),state[0],state[1],state[2],state[3])
        x = x + x_
    x_,state_ffn = self.ffn(self.ln2(x),state[4])
    x = x + x_

    if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
        xx = self.tiny_ln(x)
        q = self.tiny_q(xx)[:, :T, :]
        k = self.tiny_k(xx)[:, :T, :]
        c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
        c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
        x = x + c @ self.tiny_v(x_emb)
    return x,x_x,state_kv,x_x_rev,state_kv_rev,state_ffn

def bi_att_forward(self, x,state_xx,state_kv,state_xx_rev,state_kv_rev):
        is_last_chunk = self.args.is_last_chunk
        T, C = x.size()
        H = self.n_head
        xx = x[-1,:]
        xx_rev = x[0,:] # we ignore the xx and xx_rev because they are not used in the future forward
        r, k, v, g, w = self.jit_func(x,state_xx)
        # if is_last_chunk:
        #     #only reverse the first T-1 elements
        #     x_rev = torch.flip(x[:T-1,:], [0])
        #     x_rev = torch.cat([x_rev, x[-1,:].unsqueeze(0)],dim=0)
        # else:
        #     #reverse all elements
        #     x_rev = torch.flip(x, [0])
        # rev_r, rev_k, rev_v, rev_g, rev_w = self.jit_func(x_rev,state_xx_rev)
        #only k,v need to be reversed
        if is_last_chunk:
            k_rev = torch.flip(k[:T-1,:], [0])
            k_rev = torch.cat([k_rev, k[-1,:].unsqueeze(0)],dim=0)
            v_rev = torch.flip(v[:T-1,:], [0])
            v_rev = torch.cat([v_rev, v[-1,:].unsqueeze(0)],dim=0)
            w_rev = torch.flip(w[:T-1,:], [0])
            w_rev = torch.cat([w_rev, w[-1,:].unsqueeze(0)],dim=0)
        else:
            k_rev = torch.flip(k, [0])
            v_rev = torch.flip(v, [0])
            w_rev = torch.flip(w, [0])
        x,s = RUN_RWKV_6(1, T, C, H, state_kv.transpose(-1,-2).contiguous(),r, k, v, w, self.time_faaaa)
        x_rev,s_rev = RUN_RWKV_6(1, T, C, H, state_kv_rev.transpose(-1,-2).contiguous(),r, k_rev, v_rev, w_rev, self.time_faaaa)
        s = s.transpose(-1,-2)
        x = self.jit_func_2(x, g).squeeze(0)
        s_rev = s_rev.transpose(-1,-2)
        x_rev = self.jit_func_2(x_rev, g).squeeze(0)
        #reverse x_rev
        if is_last_chunk:
            x_rev = torch.flip(x_rev[:T-1,:], [0])
            x_rev = torch.cat([x_rev, x_rev[-1,:].unsqueeze(0)],dim=0)
        else:
            x_rev = torch.flip(x_rev, [0])
        x = x + x_rev
        return x,xx,s,xx_rev,s_rev
        
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
            args.bi_rwkv = True
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        if args.bi_rwkv:
            Block.forward = bi_block_forward
            RWKV_Tmix_x060.forward = bi_att_forward
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.emb_id = args.emb_id

    
    def forward(self, idx,state=None):
        with torch.no_grad():
            args = self.args
            T = idx.size()[0]
            is_last_chunk = True if idx[-1] == self.emb_id else False
            args.is_last_chunk = is_last_chunk
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
            NUM_STATES = 5
            if state is None:
                state = [None] * args.n_layer * NUM_STATES
                for i in range(args.n_layer): # state: 0=att_xx 1=att_kv 2=att_xx_rev 3=att_kv_rev 4=ffn_xx
                    state[i*NUM_STATES+0] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*NUM_STATES+1] = torch.zeros((args.n_head, args.n_att//args.n_head, args.n_att//args.n_head), dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*NUM_STATES+2] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*NUM_STATES+3] = torch.zeros((args.n_head, args.n_att//args.n_head, args.n_att//args.n_head), dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*NUM_STATES+4] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
            x = self.emb(idx)
            x_emb = x
            layer_id = 0
            if args.tiny_att_dim > 0:
                for block in self.blocks:
                    x,x_x,state_kv,x_x_rev,state_kv_rev,state_ffn = block(x,state[layer_id*NUM_STATES:layer_id*NUM_STATES+NUM_STATES], x_emb)
                    state[layer_id*NUM_STATES+0] = x_x
                    state[layer_id*NUM_STATES+1] = state_kv
                    state[layer_id*NUM_STATES+2] = x_x_rev
                    state[layer_id*NUM_STATES+3] = state_kv_rev
                    state[layer_id*NUM_STATES+4] = state_ffn
                    layer_id += 1
            else:
                for block in self.blocks:
                    x,x_x,state_kv,x_x_rev,state_kv_rev,state_ffn = block(x,state[layer_id*NUM_STATES:layer_id*NUM_STATES+NUM_STATES])
                    state[layer_id*NUM_STATES+0] = x_x
                    state[layer_id*NUM_STATES+1] = state_kv
                    state[layer_id*NUM_STATES+2] = x_x_rev
                    state[layer_id*NUM_STATES+3] = state_kv_rev
                    state[layer_id*NUM_STATES+4] = state_ffn
                    layer_id += 1
            x = self.ln_out(x)
            return x[-1,:],state

class RwkvForSequenceEmbedding(torch.nn.Module):

    def __init__(self, rwkvModel,embedding_id = 1, pad_id = 0,should_delete_head = True,pooling_type='weightedmean',add_mlp = False,output_dim = 0):
        super(RwkvForSequenceEmbedding, self).__init__()
        self.pad_id = pad_id
        self.rwkvModel = rwkvModel
        self.embedding_id = embedding_id
        self.pooling_type = pooling_type
        self.add_mlp = add_mlp
        if add_mlp:
            if output_dim == 0:
                output_dim = rwkvModel.args.n_embd
            self.dense = nn.Linear(rwkvModel.args.n_embd, output_dim)
            self.activation = nn.Tanh()
        if should_delete_head and hasattr(self.rwkvModel, 'head'):
            del self.rwkvModel.head

    def pooling(self, x,actual_len):
        if self.pooling_type == 'weightedmean':
            actual_len = actual_len +1
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
    def forward(self, idx,state=None):
        with torch.no_grad():
            args = self.rwkvModel.args
            T = idx.size()[0]
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
            if state is None:
                state = [None] * args.n_layer * 3
                for i in range(args.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
                    state[i*3+0] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*3+1] = torch.zeros((args.n_head, args.n_att//args.n_head, args.n_att//args.n_head), dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*3+2] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
            x = self.rwkvModel.emb(idx)
            x_emb = x
            layer_id = 0
            if args.tiny_att_dim > 0:
                for block in self.rwkvModel.blocks:
                    x,x_x,state_kv,state_ffn = block(x,state[layer_id*3:layer_id*3+3], x_emb)
                    state[layer_id*3+0] = x_x
                    state[layer_id*3+1] = state_kv
                    state[layer_id*3+2] = state_ffn
                    layer_id += 1
            else:
                for block in self.rwkvModel.blocks:
                    x,x_x,state_kv,state_ffn = block(x,state[layer_id*3:layer_id*3+3])
                    state[layer_id*3+0] = x_x
                    state[layer_id*3+1] = state_kv
                    state[layer_id*3+2] = state_ffn
                    layer_id += 1

            x = self.rwkvModel.ln_out(x)

            #calculate the idx actual length which is first self.embedding_id
            idx_actual_len = torch.tensor([T-1],device = idx.device,dtype=torch.long)
            x = self.pooling(x.unsqueeze(0),idx_actual_len)
            if self.add_mlp:
                x = self.activation(self.dense(x))
            return x.squeeze(0),state
    
class RwkvForClassification(torch.nn.Module):

    def __init__(self, rwkvModel, num_labels=1,class_id = 1, pad_id = 0,should_delete_head = True):
        super(RwkvForClassification, self).__init__()
        self.pad_id = pad_id
        self.class_id = class_id
        self.rwkvModel = rwkvModel
        if should_delete_head and hasattr(self.rwkvModel, 'head'):
            del self.rwkvModel.head
        self.score = nn.Linear(rwkvModel.args.n_embd, num_labels,bias=False)
        self.num_labels = num_labels
    def forward(self, idx,state = None):
        with torch.no_grad():
            args = self.rwkvModel.args
            T = idx.size()[0]
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
            if state is None:
                state = [None] * args.n_layer * 3
                for i in range(args.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
                    state[i*3+0] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*3+1] = torch.zeros((args.n_head, args.n_att//args.n_head, args.n_att//args.n_head), dtype=torch.float, requires_grad=False, device=idx.device).contiguous()
                    state[i*3+2] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=idx.device).contiguous()

            x = self.rwkvModel.emb(idx)
            x_emb = x
            layer_id = 0

            if args.tiny_att_dim > 0:
                for block in self.rwkvModel.blocks:
                    x,x_x,state_kv,state_ffn = block(x,state[layer_id*3:layer_id*3+3], x_emb)
                    state[layer_id*3+0] = x_x
                    state[layer_id*3+1] = state_kv
                    state[layer_id*3+2] = state_ffn
                    layer_id += 1
            else:
                for block in self.rwkvModel.blocks:
                    x,x_x,state_kv,state_ffn = block(x,state[layer_id*3:layer_id*3+3])
                    state[layer_id*3+0] = x_x
                    state[layer_id*3+1] = state_kv
                    state[layer_id*3+2] = state_ffn
                    layer_id += 1

            x = self.rwkvModel.ln_out(x)

            logits = self.score(x)
            pooled_logits = logits[-1]
            return pooled_logits,state

def set_adapter(rwkv, adapter_name: str | list[str]) -> None:
    """Set the active adapter(s).

    Args:
        adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
    """
    for module in rwkv.modules():
        if isinstance(module, LoraLayer):
            module._active_adapter = adapter_name
            module._disable_adapters = False

def enable_lora(rwkv, enable=True) -> None:
    for module in rwkv.modules():
        if isinstance(module, LoraLayer):
            module._disable_adapters = not enable

class BiEncoder:
    def __init__(self,
                    rwkv,
                    lora_path,
                    tokenizer,
                    dtype=torch.float,
                    device='cuda',
                    lora_type='lora',
                    add_mlp=True,
                    mlp_dim=1024,
                    lora_r=8,
                    lora_alpha=32,
                    target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                    adapter_name='bi_embedding_lora',
                    original_adapter_name='embedding_lora',
                    should_delete_head = False) -> None:
        self.rwkv = rwkv
        self.add_mlp = add_mlp
        self.mlp_dim = mlp_dim
        args = rwkv.args

        self.lora_path = lora_path
        lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        from peft import inject_adapter_in_model
        rwkv = inject_adapter_in_model(lora_config,rwkv,adapter_name=adapter_name)
        self.adapter_name = adapter_name
        print(f'inject lora from {lora_path} to model,result is {rwkv}')

        self.rwkv_embedding = RwkvForSequenceEmbedding(rwkv,add_mlp=add_mlp,output_dim=mlp_dim,should_delete_head=should_delete_head)
        print(self.rwkv_embedding)
        with torch.no_grad():
            w = torch.load(lora_path)
            #replace keys with original adapter name to new adapter name
            if original_adapter_name != adapter_name:
                print(f'origal_keys:{list(w.keys())}')
                for k in list(w.keys()):
                    if original_adapter_name in k:
                        new_k = k.replace(original_adapter_name,adapter_name)
                        w[new_k] = w.pop(k)
            info = self.rwkv_embedding.load_state_dict(w,strict=False)
            print(f'load model from {lora_path},result is {info}')
        self.rwkv_embedding = self.rwkv_embedding.to(dtype)
        self.rwkv_embedding = self.rwkv_embedding.to(device)
        self.rwkv_embedding.eval()
        self.dtype = dtype
        self.device = device
        self.tokenizer = tokenizer

    def encode_texts(self,text,chunk_size=1024):
        set_adapter(self.rwkv,self.adapter_name)
        MAX_LEN = 4096
        max_len = 0
        input_ids =  self.tokenizer.encode(text)
        input_ids.append(self.rwkv_embedding.embedding_id)
        state = None
        offset = 0
        while offset < len(input_ids):
            chunk = input_ids[offset:offset+chunk_size]
            with torch.autocast(enabled=True,device_type='cuda',dtype=self.dtype):
                outputs,state = self.rwkv_embedding(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
            offset += len(chunk)

        return outputs


class CrossEncoder:
    def __init__(self,
                    rwkv,
                    lora_path,
                    tokenizer,
                    dtype=torch.float,
                    device='cuda',
                    lora_type='lora',
                    lora_r=8,
                    lora_alpha=32,
                    target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                    adapter_name='cross_encoder_lora',
                    original_adapter_name='embedding_lora',
                    sep_token_id = 2,
                    should_delete_head = False) -> None:
        self.rwkv = rwkv
        self.lora_path = lora_path
        lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        from peft import inject_adapter_in_model
        rwkv = inject_adapter_in_model(lora_config,rwkv,adapter_name=adapter_name)
        print(f'inject lora from {lora_path} to model,result is {rwkv}')
        self.adapter_name = adapter_name
        self.cross_encoder = RwkvForClassification(rwkv,should_delete_head=should_delete_head)
        with torch.no_grad():
            w = torch.load(lora_path)
            #replace keys with original adapter name to new adapter name
            if original_adapter_name != adapter_name:
                print(f'origal_keys:{list(w.keys())}')
                for k in list(w.keys()):
                    if original_adapter_name in k:
                        new_k = k.replace(original_adapter_name,adapter_name)
                        w[new_k] = w.pop(k)
            info = self.cross_encoder.load_state_dict(w,strict=False)
            print(f'load model from {lora_path},result is {info}')
        self.cross_encoder = self.cross_encoder.to(dtype)
        self.cross_encoder = self.cross_encoder.to(device)
        self.cross_encoder.eval()
        self.dtype = dtype
        self.device = device
        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id

    def encode_texts(self,text_a, text_b,chunk_size=128):
        set_adapter(self.rwkv,self.adapter_name)
        MAX_LEN = 4096
        max_len = 0
        text_a_ids = self.tokenizer.encode(text_a)
        text_b_ids = self.tokenizer.encode(text_b)
        all_input_ids = text_a_ids + [self.sep_token_id]+text_b_ids+[self.cross_encoder.class_id]
        offset = 0
        state = None
        while offset < len(all_input_ids):
            chunk = all_input_ids[offset:offset+chunk_size]
            with torch.autocast(enabled=True,device_type='cuda',dtype=self.dtype):
                outputs,state = self.cross_encoder(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
            offset += len(chunk)
        return outputs

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
            args.n_head = n_head
            args.n_att = dim_att
            return w
    except Exception as e:
        import traceback
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

class BiCrossFusionEncoder:
    """
    This encoder is to fuse the bi-encoder and cross-encoder with the same rwkv base model injected with 2 sets of lora adapters.
    We have 2 assumptions here:
    1. The bi-encoder and cross-encoder share the same base model
    2. The lora types of bi-encoder and cross-encoder are the same
    This class instance is not thread-safe since we need to switch the adapter name before encoding.
    """
    def __init__(self,
                 rwkv,
                 bi_lora_path,
                 cross_lora_path,
                 tokenizer,
                 dtype=torch.float,
                 device='cuda',
                 lora_type='lora',
                 lora_r=8,
                 lora_alpha=32,
                 add_mlp=True,
                 mlp_dim=1024,
                 target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                 cross_adapter_name='cross_encoder_lora',
                 original_cross_adapter_name='embedding_lora',
                 bi_adapter_name='bi_embedding_lora',
                 original_bi_adapter_name='embedding_lora',
                 sep_token_id = 2,
                 should_delete_head = False
                 ) -> None:
        #load cross encoder and inject cross adapter
        cross_lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            cross_lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            cross_lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        from peft import inject_adapter_in_model
        rwkv = inject_adapter_in_model(cross_lora_config,rwkv,adapter_name=cross_adapter_name)

        #load bi encoder and inject bi adapter
        bi_lora_config = None
        if lora_type == 'lora':
            from peft import LoraConfig
            bi_lora_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        elif lora_type == 'adalora':
            from peft import AdaLoraConfig
            bi_lora_config = AdaLoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=target_modules,lora_dropout=0)
        rwkv = inject_adapter_in_model(bi_lora_config,rwkv,adapter_name=bi_adapter_name)

        self.cross_encoder = RwkvForClassification(rwkv,should_delete_head=should_delete_head)
        self.bi_encoder = RwkvForSequenceEmbedding(rwkv,add_mlp=add_mlp,output_dim=mlp_dim,should_delete_head=should_delete_head)

        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id

        #load cross encoder lora params
        with torch.no_grad():
            w = torch.load(cross_lora_path)
            #replace keys with original adapter name to new adapter name
            if original_cross_adapter_name != cross_adapter_name:
                print(f'origal_keys:{list(w.keys())}')
                for k in list(w.keys()):
                    if original_cross_adapter_name in k:
                        new_k = k.replace(original_cross_adapter_name,cross_adapter_name)
                        w[new_k] = w.pop(k)
            info = self.cross_encoder.load_state_dict(w,strict=False)
            print(f'load model from {cross_lora_path},result is {info}')
        self.cross_encoder = self.cross_encoder.to(dtype)
        self.cross_encoder = self.cross_encoder.to(device)
        self.cross_encoder.eval()

        #load bi encoder lora params
        with torch.no_grad():
            w = torch.load(bi_lora_path)
            #replace keys with original adapter name to new adapter name
            if original_bi_adapter_name != bi_adapter_name:
                print(f'origal_keys:{list(w.keys())}')
                for k in list(w.keys()):
                    if original_bi_adapter_name in k:
                        new_k = k.replace(original_bi_adapter_name,bi_adapter_name)
                        w[new_k] = w.pop(k)
            info = self.bi_encoder.load_state_dict(w,strict=False)
            print(f'load model from {bi_lora_path},result is {info}')
        self.bi_encoder = self.bi_encoder.to(dtype)
        self.bi_encoder = self.bi_encoder.to(device)
        self.bi_encoder.eval()

        self.bi_adapter_name = bi_adapter_name
        self.cross_adapter_name = cross_adapter_name
        self.rwkv = rwkv
        self.dtype = dtype
        self.device = device

        print(f'inject lora from {cross_lora_path} and {bi_lora_path} to model,result is {rwkv}')


    def encode_texts(self,text,chunk_size=1024):
        set_adapter(self.rwkv,self.bi_adapter_name)
        input_ids =  self.tokenizer.encode(text)
        input_ids.append(self.bi_encoder.embedding_id)
        state = None
        offset = 0
        while offset < len(input_ids):
            chunk = input_ids[offset:offset+chunk_size]
            with torch.autocast(enabled=True,device_type='cuda',dtype=self.dtype):
                outputs,state = self.bi_encoder(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
            offset += len(chunk)

        return outputs
    
    def cross_encode_texts(self,text_a, text_b,chunk_size=1024):
        set_adapter(self.rwkv,self.cross_adapter_name)
        text_a_ids = self.tokenizer.encode(text_a)
        text_b_ids = self.tokenizer.encode(text_b)
        all_input_ids = text_a_ids + [self.sep_token_id]+text_b_ids+[self.cross_encoder.class_id]
        offset = 0
        state = None
        while offset < len(all_input_ids):
            chunk = all_input_ids[offset:offset+chunk_size]
            with torch.autocast(enabled=True,device_type='cuda',dtype=self.dtype):
                outputs,state = self.cross_encoder(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
            offset += len(chunk)
        return outputs

import numpy as np
from torch.nn import functional as F


def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    # 'privateuseone' is the type of custom devices like `torch_directml.device()`
    if probs.device.type in ['cpu', 'privateuseone']:
        probs = probs.cpu().numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return int(out)
    else:
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
    
from rwkv.utils import PIPELINE_ARGS
from src.logits_processors import TopKLogitsProcess,TopPLogitsProcess,RepetitionPenaltyLogitsProcessor
def generate(model, ctx,tokenizer, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None,device='cuda',repetition_penalty=1.0):
    logits_warpper = []
    logits_processes = []
    logits_warpper.append(TopPLogitsProcess(top_p=args.top_p,min_tokens_to_keep=args.top_k if args.top_k > 0 else 1))
    if args.top_k > 0:
        logits_warpper.append(TopKLogitsProcess(top_k=args.top_k))
    logits_processes.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    sep_id = 1
    for i in range(token_count):
        # forward & adjust prob.
        # tokens = tokenizer.encode(ctx) + [sep_id] if i == 0 else [token]
        tokens = tokenizer.encode(ctx) if i == 0 else [token]
        tokens = torch.tensor(tokens, dtype=torch.long,device=device)
        while len(tokens) > 0:
            out, state = model.forward(tokens[:args.chunk_len], state)
            tokens = tokens[args.chunk_len:]
            
        for n in args.token_ban:
            out[n] = -float('inf')
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
        for process in logits_warpper:
            out = process.process(torch.tensor(all_tokens,dtype=torch.long,device=out.device), out)
        for process in logits_processes:
            out = process.process(torch.tensor(all_tokens,dtype=torch.long,device=out.device), out)
        probs = F.softmax(out, dim=-1)
        token =  int(torch.multinomial(probs, num_samples=1)[0])
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= args.alpha_decay
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        # print(occurrence) # debug
        
        # output
        if callback:
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                callback(tmp)
                out_str += tmp
                out_last = i + 1
    out_str = tokenizer.decode(all_tokens)
    return out_str   

class BeamHypothesis:

    def __init__(self,
                 num_beams: int, 
                 length_penalty: float, 
                 early_stopping: bool, 
                 max_length: Optional[int] = None) -> None:
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )
    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(
        self,
        hyp: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: Optional[torch.LongTensor] = None,
        generated_len: Optional[int] = None,
    ):
        """
        Add a new hypothesis to the list.
        """
        if generated_len is not None:
            score = sum_logprobs / (generated_len**self.length_penalty)
        # This 'else' case exists for retrocompatibility
        else:
            score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int] = 0) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
            # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                if self.max_length <= decoder_prompt_len:
                    raise ValueError("max_length is not larger than decoder prompt length")
                highest_attainable_score = (
                    best_sum_logprobs / (self.max_length - decoder_prompt_len) ** self.length_penalty
                )
            # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
            else:
                highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret


def clone_state(state):
    if state is None:
        return None
    return [s.clone() for s in state]

def generate_beamsearch(model, 
                        ctx,
                        tokenizer, 
                        token_count=100, 
                        callback=None, 
                        state=None,
                        num_beams=10,
                        return_num_sequences=5,
                        eos_id = [0,1],
                        block_size = 512,
                        device='cuda',
                        repetition_penalty = 1.5,
                        do_sample = True,
                        num_group = 5,
                        length_penalty = 0.5,
                        is_sum_logprobs = False,
                        top_p = 0.96,
                        top_k = 30,
                        ):
    logits_warpper = []
    logits_processes = []
    if do_sample:
        logits_warpper.append(TopPLogitsProcess(top_p=top_p,min_tokens_to_keep=top_k if top_k > 0 else 1))
        if top_k > 0:
            logits_warpper.append(TopKLogitsProcess(top_k=top_k))
    logits_processes.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    input_ids = torch.tensor(
        tokenizer.encode(ctx), dtype=torch.long, device=device)
    group_size = num_beams // num_group
    hypothesis = [BeamHypothesis(num_beams, length_penalty, False,token_count) for _ in range(num_group)]
    is_initalized = False
    inputs = []
    reserve_beam_size = max(2,1+len(eos_id))*num_beams
    groups_done = [False for _ in range(num_group)]
    for step in range(token_count):
        results = []
        if not is_initalized:
            #initialize the output and state
            idx = 0
            ctx_len = len(input_ids)
            #create initial state
            while idx < ctx_len:
                out, state = model.forward(input_ids[idx:idx+block_size], state)
                idx += block_size
            is_initalized = True
            results.append((torch.tensor([],dtype=torch.long,device=device),0.0,out,clone_state(state),None))
            #now we can start to do beam search here with empty input_ids
        for input_ids,logprob, old_state,beam_idx in inputs:
            out, new_state = model.forward(input_ids[-1:], clone_state(old_state))
            results.append((input_ids,logprob,out,new_state,beam_idx))
        for _,_,state,_ in inputs:
            del state
        inputs = []
        for idx, (input_ids,logprob,out,state,beam_idx) in enumerate(results):
            next_token_scores = F.log_softmax(out,dim=-1)
            if is_sum_logprobs:
                next_token_scores = next_token_scores+logprob
            for logits_process in logits_processes:
                next_token_scores = logits_process.process(input_ids,next_token_scores)

            if do_sample:
                for logits_warper in logits_warpper:
                    next_token_scores = logits_warper.process(input_ids,next_token_scores)
                #check if all are torch.tensor(float('-inf'))
                probs = F.softmax(next_token_scores,dim=-1)
                next_tokens = torch.multinomial(probs,num_samples=reserve_beam_size)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=0)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, reserve_beam_size, dim=0, largest=True, sorted=True
                )
            #handle the search results
            added_cnt = 0
            for token_rank,(token,score) in enumerate(zip(next_tokens,next_token_scores)):
                if score == float('-inf'):
                    continue
                new_score = score 
                if token in eos_id and token_rank < group_size:
                    hyp_group_idx = token_rank if beam_idx is None else beam_idx // group_size
                    hypothesis[hyp_group_idx].add(input_ids,new_score,beam_idx if beam_idx is not None else token_rank,step+1)
                else:
                    new_input_ids = torch.cat([input_ids,token.unsqueeze(0)])
                    inputs.append((new_input_ids,new_score,state,beam_idx if beam_idx is not None else token_rank))
                    added_cnt += 1
                
                if (added_cnt >= group_size and beam_idx is not None) or (added_cnt >= num_beams and beam_idx is None): 
                    if beam_idx is not None:
                        hyp_group_idx = beam_idx // group_size
                        if hypothesis[hyp_group_idx].is_done(next_token_scores.max().item(),step+1):
                            groups_done[hyp_group_idx] = True
                    break
        #check if we should stop
        should_stop = sum(groups_done)
        if should_stop == num_group:
            break
    #return the top k sequences
    #handle the open searches
    outputs = []
    for idx, (input_ids,logprob,state,beam_idx) in enumerate(inputs):
        outputs.append((logprob/(step**length_penalty),input_ids,beam_idx))
    for hyp in hypothesis:
        outputs += hyp.beams
    outputs = sorted(outputs,key=lambda x:x[0].item(),reverse=True)
    outputs = outputs
    return outputs[:return_num_sequences]

    # while idx < ctx_len:
    #     out, state = model.forward(input_ids[idx:idx+block_size], state)
    #     idx += block_size
    # reserve_beam_size = max(2,len(eos_id))*num_beams
    # beam_hypotheses = BeamHypothesis(num_beams=reserve_beam_size, length_penalty=1.0, early_stopping=False)
    # next_token_scores = F.log_softmax(out,dim=-1)#size s (vocab_size)
    # beam_search = [(torch.tensor([],dtype=torch.long,device=device), 0.0,next_token_scores, state)]
    # for step in range(token_count):
    #     for beam in beam_search:
    #         input_ids, sum_logprobs, next_token_scores, state = beam
    #         do_repetition_penalty(input_ids, next_token_scores, repetition_penalty)
    #         if do_sample:
    #             probs = F.softmax(next_token_scores,dim=-1)#size s (vocab_size)
    #             next_tokens = torch.multinomial(probs,
    #                                             num_samples=reserve_beam_size)#size is (reserve_beam_size)
    #             next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
    #             next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
    #             next_tokens = torch.gather(next_tokens, -1, _indices)
    #         else:
    #             next_token_scores, next_tokens = torch.topk(
    #                 next_token_scores, reserve_beam_size, dim=1, largest=True, sorted=True
    #             )
            

gen_cnt = 0


def my_print(s):
    global gen_cnt
    gen_cnt += 1
    print(s, end='', flush=True)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    ckpt = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    device = 'cuda'
    dtype = torch.bfloat16
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(ckpt, args)
    print(args)
    model = RWKV(args)
    info = model.load_state_dict(w)
    model.eval()
    print(model)
    print(info)
    gen_args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.8, top_k = 100, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,2], # stop generation whenever you see any token here
                        chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
    tokenizer_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    ctx = '你是一个编程助手，我会向你提出需求，你需要根据需求编写代码。\nBot:好的，请提出需求。\nUser：编写一个函数，输入一个Shape是(B,T,C)的张量，把一个全零的(C)张量扩展成(B,1,C)，和输入张量相加，最后变成(B,T+1,C)。请用PyTorch实现。\nBot:好的，请稍等。\n'
    print(tokenizer.encode(ctx))
    model = model.to(dtype)
    model = model.to(device)
    with torch.no_grad():
        with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
            output = generate(model, ctx,tokenizer, token_count=512, args=gen_args,callback=my_print)
        print(output)

    bi_lora_path = '/media/yueyulin/data_4t/models/lora/biencoder/epoch_1_step_430000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    cross_lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/cross-encoder/epoch_0_step_500000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    fusedEncoder = BiCrossFusionEncoder(model,bi_lora_path,cross_lora_path,tokenizer,dtype=dtype,lora_type='lora',lora_r=8,lora_alpha=32,add_mlp=True,mlp_dim=1024,target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],cross_adapter_name='cross_encoder_lora',original_cross_adapter_name='embedding_lora',bi_adapter_name='bi_embedding_lora',original_bi_adapter_name='embedding_lora',sep_token_id = 2)
    print(fusedEncoder.bi_encoder)
    print(fusedEncoder.cross_encoder)

    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = [fusedEncoder.encode_texts(text) for text in texts]
    print(outputs)
    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')
    enable_lora(model,enable=False)
    with torch.no_grad():
        with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
            output = generate(model, ctx,tokenizer, token_count=512, args=gen_args,callback=my_print)
        print(output)

    out  = fusedEncoder.cross_encode_texts(texts[0],"北京是中华人民共和国不可分割的一部分")
    print(out)

    out = fusedEncoder.cross_encode_texts(texts[0],output)
    print(out)

    out = fusedEncoder.cross_encode_texts(texts[0],texts[1])
    print(out)
    out = fusedEncoder.cross_encode_texts(texts[0],texts[2])
    print(out)
    out = fusedEncoder.cross_encode_texts(texts[0],texts[3])
    print(out)
    out = fusedEncoder.cross_encode_texts(texts[2],texts[3])
    print(out)
    print(args)
if __name__ == '__main__1':
    torch.backends.cudnn.benchmark = True
    ckpt = '/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    device = 'cuda'
    dtype = torch.bfloat16
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(ckpt, args)
    print(args)
    model = RWKV(args)
    info = model.load_state_dict(w)
    model.eval()
    print(info)

    gen_args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.8, top_k = 100, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,2], # stop generation whenever you see any token here
                        chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
    tokenizer_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    ctx = '你是一个编程助手，我会向你提出需求，你需要根据需求编写代码。\nBot:好的，请提出需求。\nUser：编写一个函数，输入一个Shape是(B,T,C)的张量，把一个全零的(C)张量扩展成(B,1,C)，和输入张量相加，最后变成(B,T+1,C)。请用PyTorch实现。\nBot:好的，请稍等。\n'
    print(tokenizer.encode(ctx))
    model = model.to(dtype)
    model = model.to(device)
    with torch.no_grad():
        with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
            output = generate(model, ctx,tokenizer, token_count=512, args=gen_args,callback=my_print)
        print(output)

    lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/bi-encoder/add_mlp_in_batch_neg/epoch_0_step_200000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    bi_encoder = BiEncoder(model,lora_path,tokenizer,dtype=dtype,lora_type='lora',add_mlp=True,mlp_dim=1024,lora_r=8,lora_alpha=32,target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],adapter_name='bi_embedding_lora',original_adapter_name='embedding_lora')
    print(bi_encoder)
    embeddings = bi_encoder.encode_texts(output)
    print(embeddings)
    print(embeddings.shape)

    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = [bi_encoder.encode_texts(text) for text in texts]


    print(outputs)
    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')

   
    cross_lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/cross-encoder/epoch_0_step_500000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    cross_encoder = CrossEncoder(model,cross_lora_path,tokenizer,dtype=dtype,lora_type='lora',lora_r=8,lora_alpha=32,target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],adapter_name='cross_encoder_lora',original_adapter_name='embedding_lora',sep_token_id = 2)
    print(cross_encoder)
    out  = cross_encoder.encode_texts(texts[0],texts[1])
    print(out)

    enable_lora(model,enable=False)
    with torch.no_grad():
        with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
            output = generate(model, ctx,tokenizer, token_count=512, args=gen_args,callback=my_print)
        print(output)

    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = [bi_encoder.encode_texts(text) for text in texts]


    print(outputs)
    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')

    enable_lora(model,enable=False)
    with torch.no_grad():
        with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
            output = generate(model, ctx,tokenizer, token_count=512, args=gen_args,callback=my_print)
        print(output)

    out  = cross_encoder.encode_texts(texts[0],"北京是中华人民共和国不可分割的一部分")
    print(out)

    out = cross_encoder.encode_texts(texts[0],output)
    print(out)
    print(args)