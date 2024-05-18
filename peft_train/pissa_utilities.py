if __name__ == '__main__':
    from peft import LoraConfig
    import argparse
    import torch
    parser = argparse.ArgumentParser()
        #add peft arguments
    parser.add_argument('--lora_type', type=str, default='lora', help='lora type', choices=['lora','adalora'])
    parser.add_argument('--target_modules', type=str, nargs='+',default=['emb','ffn.key','ffn.value','ffn.receptance','att.key','att.value','att.receptance'], help='target modules')
    parser.add_argument('--lora_r',type=int,default=8)
    parser.add_argument('--lora_alpha',type=int,default=32)
    parser.add_argument('--lora_dropout',type=float,default=0.1)
    args = parser.parse_args()
    lora_config = LoraConfig(
            # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
            init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
            r=args.lora_r,lora_alpha=args.lora_alpha,target_modules=args.target_modules,lora_dropout=args.lora_dropout
        )
    
    ckpt = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    import os
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
    from src.model_run import RWKV,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate
    device = 'cuda'
    dtype = torch.bfloat16
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(ckpt, args)
    print(args)
    model = RWKV(args)
    info = model.load_state_dict(w)
    model.eval()
    from peft import inject_adapter_in_model
    model = inject_adapter_in_model(lora_config,model,adapter_name='sft_lora')
    for name, param in model.named_parameters():
        print(name)