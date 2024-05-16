import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_filename', type=str,default='/media/yueyulin/data_4t/models/states_tuning/epoch_0_step_1000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth')
    parser.add_argument('--output_file', type=str,default='/tmp/states.pth')

    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = os.path.basename(args.model_filename) + '_states.pth'
    import torch
    w = torch.load(args.model_filename,map_location='cpu')
    trainable_params = {}
    #select the parameters which contains 'time_state'
    for name, param in w.items():
        if 'time_state' in name:
            trainable_params[name] = param
            print(name)
            for p in param:
                print(p.shape)
            print('-------------------')
    torch.save(trainable_params, args.output_file)
    