import torch

HEAD_SIZE = 64

import torch.nn.functional as F

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

def optimized_pytorch_forward_xx(B, T, C, H, r, k, v, w, u):
    assert HEAD_SIZE == C // H
    assert r.dtype == torch.float32
    assert k.dtype == torch.float32
    assert v.dtype == torch.float32
    assert w.dtype == torch.float32
    assert u.dtype == torch.float32

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

def optimized_pytorch_forward_x(B, T, C, H, r, k, v, w, u):
    assert HEAD_SIZE == C // H

    # 重塑张量并转换为float32
    r = r.view(B, T, H, HEAD_SIZE)
    k = k.view(B, T, H, HEAD_SIZE)
    v = v.view(B, T, H, HEAD_SIZE)
    w = w.view(B, T, H, HEAD_SIZE)

    # 处理w，匹配CUDA实现
    ew = torch.exp(-torch.exp(w))

    # 处理u
    u = u.float()

    y = torch.zeros((B, T, H, HEAD_SIZE), device=r.device, dtype=torch.float32)
    
    # 初始化状态张量，现在包含所有batch
    state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, dtype=torch.float32, device=r.device)

    for h in range(H):
        uh = u[h]
        for t in range(T):
            rt = r[:, t, h]  # Shape: [B, HEAD_SIZE]
            kt = k[:, t, h]  # Shape: [B, HEAD_SIZE]
            vt = v[:, t, h]  # Shape: [B, HEAD_SIZE]
            wt = ew[:, t, h]  # Shape: [B, HEAD_SIZE]
            
            for j in range(HEAD_SIZE):
                v_h = vt[:, j].unsqueeze(1)  # Shape: [B, 1]
                x = kt * v_h  # Shape: [B, HEAD_SIZE]
                y_local = torch.sum(rt * (x * uh + state[:, h, j]), dim=1)  # Shape: [B]
                state[:, h, j] = state[:, h, j] * wt + x
                y[:, t, h, j] = y_local

    return y.view(B, T, C)

def optimized_pytorch_forward(B, T, C, H, r, k, v, w, u):
    assert HEAD_SIZE == C // H
    assert r.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert w.dtype == torch.bfloat16
    assert u.dtype == torch.bfloat16

    # 重塑张量并转换为float32
    r = r.view(B, T, H, HEAD_SIZE).float()
    k = k.view(B, T, H, HEAD_SIZE).float()
    v = v.view(B, T, H, HEAD_SIZE).float()
    w = w.view(B, T, H, HEAD_SIZE).float()

    # 处理w，匹配CUDA实现
    ew = torch.exp(-torch.exp(w))

    # 处理u
    u = u.float()

    y = torch.zeros((B, T, H, HEAD_SIZE), device=r.device, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            state = torch.zeros(HEAD_SIZE, HEAD_SIZE, dtype=torch.float32, device=r.device)
            uh = u[h]
            for t in range(T):
                rt = r[b, t, h]
                kt = k[b, t, h]
                vt = v[b, t, h]
                wt = ew[b, t, h]
                
                for j in range(HEAD_SIZE):
                    v_h = vt[j]
                    x = kt * v_h
                    y_local = torch.sum(rt * (x * uh + state[j]))
                    state[j] = state[j] * wt + x
                    y[b, t, h, j] = y_local

    return y.bfloat16().view(B, T, C)

def pytorch_forward(B, T, C, H, r, k, v, w, u):
    assert HEAD_SIZE == C // H
    
    # 重塑张量
    r = r.view(B, T, H, HEAD_SIZE)
    k = k.view(B, T, H, HEAD_SIZE)
    v = v.view(B, T, H, HEAD_SIZE)
    w = w.view(B, T, H, HEAD_SIZE)
    
    # 处理w，匹配CUDA实现
    ew = torch.exp((-torch.exp(w)))
    
    # 初始化y和state
    y = torch.empty((B, T, H,HEAD_SIZE), device=r.device, dtype=torch.float32)
    # state = torch.zeros(B, H, HEAD_SIZE,HEAD_SIZE, dtype=torch.float32, device=r.device)
    
    # 对u进行处理
    u = u.float().view(H, HEAD_SIZE)
    
    for b in range(B):
        for h in range(H):
            state = torch.zeros(HEAD_SIZE,HEAD_SIZE, dtype=torch.float32, device=r.device)
            uh = u[h]
            for t in range(T):
                rt = r[b, t, h]
                kt = k[b, t, h]
                vt = v[b, t, h]
                wt = ew[b, t, h]
                
                for j in range(HEAD_SIZE):
                    v_h = vt[j]
                    y_local = 0
                    for i in range(HEAD_SIZE):
                        k_h = kt[i]
                        x = k_h*v_h
                        r_h = rt[i]
                        w_h = wt[i]
                        y_local += r_h*(x*uh[i]+state[j,i])
                        state[j,i] = state[j,i]*w_h+x
                    y[b,t,h,j] = y_local             

    return y.view(B, T, C)

import os
from torch.utils.cpp_extension import load
HEAD_SIZE = 64
os.environ['RWKV_CTXLEN'] = '4096'
wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                            verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
print(wkv6_cuda)

def forward(B, T, C, H, r, k, v, w, u):
    with torch.no_grad():
        assert r.dtype == torch.bfloat16
        assert k.dtype == torch.bfloat16
        assert v.dtype == torch.bfloat16
        assert w.dtype == torch.bfloat16
        assert u.dtype == torch.bfloat16
        assert HEAD_SIZE == C // H
        assert r.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        assert w.is_contiguous()
        assert u.is_contiguous()
        ew = (-torch.exp(w.float())).contiguous()
        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
        wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
        return y

# 使用示例
B, T, C = 2, 10, 256
H = C // HEAD_SIZE

r = torch.randn(B, T, C, dtype=torch.float32)
k = torch.randn(B, T, C, dtype=torch.float32)
v = torch.randn(B, T, C, dtype=torch.float32)
w = torch.randn(B, T, C, dtype=torch.float32)
u = torch.randn(H, HEAD_SIZE, dtype=torch.float32)

result = pytorch_forward(B, T, C, H,r, k, v, w, u)
print(result.shape)
result_optimized = optimized_pytorch_forward_xx(B, T, C, H,r, k, v, w, u)
print(result_optimized.shape)
r_cuda = r.to(device='cuda',dtype=torch.bfloat16)
k_cuda = k.to(device='cuda',dtype=torch.bfloat16)
v_cuda = v.to(device='cuda',dtype=torch.bfloat16)
w_cuda = w.to(device='cuda',dtype=torch.bfloat16)
u_cuda = u.to(device='cuda',dtype=torch.bfloat16)
result_cuda = forward(B, T, C, H, r_cuda, k_cuda, v_cuda, w_cuda, u_cuda)
print(result_cuda.shape)
print(result[0][0])
print(result_cuda[0][0])
#compare result and result_cuda
print(torch.allclose(result, result_cuda.float().cpu(), atol=1e-2))
print(result_optimized[0][0])
print("结果是否接近:", torch.allclose(result, result_optimized, atol=1e-2))
print("最大误差:", torch.max(torch.abs(result - result_optimized)))

result_final = run_rwkv6_forward(r, k, v, w, u)
print("结果是否接近:", torch.allclose(result, result_final, atol=1e-2))
print("最大误差:", torch.max(torch.abs(result - result_final)))