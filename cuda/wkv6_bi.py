import os
from torch.utils.cpp_extension import load
os.environ["RWKV_HEAD_SIZE_A"]='64'
os.environ['RWKV_CTXLEN']='4096'
HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wkv6_cuda = load(name="wkv6_bi", sources=[f"{parent_dir}/cuda/wkv6_bi_op.cpp", f"{parent_dir}/cuda/wkv6_bi_cuda.cu"],
                            verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
print(wkv6_cuda)

import torch

class WKV_6_BI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H,mask, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert mask.dtype == torch.int
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            ctx.mask = mask
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            assert mask.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H,mask, r, k, v, ew, u, y)
            return y
    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            mask = ctx.mask
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H,mask, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None,None, gr, gk, gv, gw, gu)
def RUN_CUDA_RWKV6(B, T, C, H,mask, r, k, v, w, u):
    return WKV_6_BI.apply(B, T, C, H,mask, r, k, v, w, u)

B,T,C = 2,64,1024
H = 16
mask = torch.ones((B, T), device='cuda', dtype=torch.int,requires_grad=False).contiguous()
print(mask)
mask[0,60:] = 0
mask[1,40:] = 0
print(mask)
r = torch.randn((B, T, C), device='cuda', dtype=torch.bfloat16,requires_grad=True).contiguous()
k = torch.randn((B, T, C), device='cuda', dtype=torch.bfloat16,requires_grad=True).contiguous()
v = torch.randn((B, T, C), device='cuda', dtype=torch.bfloat16,requires_grad=True).contiguous()
w = torch.randn((B, T, C), device='cuda', dtype=torch.bfloat16,requires_grad=True).contiguous()
u = torch.randn((H, HEAD_SIZE), device='cuda', dtype=torch.bfloat16,requires_grad=True).contiguous()
y = RUN_CUDA_RWKV6(B, T, C, H,mask, r, k, v, w, u)
print(y[0])
print(mask[0])
print('-----------------')
print(y[1])
print(mask[1])
print('-----------------')
print(y[0,60:])
print(y[1,40:])
print('-----------------')
print(y[0,:60])
print(y[1,:40])

gy = torch.randn((B, T, C), device='cuda', dtype=torch.bfloat16).contiguous()
y.requires_grad_(True)
y.backward(gy)
print(r.grad)
print(k.grad)
print(v.grad)
print(w.grad)
print(u.grad)
