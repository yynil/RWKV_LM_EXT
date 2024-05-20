import argparse
import torch
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert HF lora to RWKV lora")
    parser.add_argument("--hf_lora", type=str, help="Input HF lora file")
    parser.add_argument("--rwkv_lora", type=str, help="Output RWKV lora file")
    args = parser.parse_args()

    hf_lora_state_dict = torch.load(args.hf_lora, map_location='cpu')
    rwkv_lora_state_dict = {}

    keys_in_hf = hf_lora_state_dict.keys()
    #check if head has a prefix
    prefix = ""
    lora_name = ""
    for key  in keys_in_hf:
        if ".blocks." in key :
            prefix = key.split(".blocks.")[0]
            weight_idx = key.rfind(".weight")
            if weight_idx != -1:
                prev_dot = key.rfind(".", 0,weight_idx)
                if prev_dot != -1:
                    lora_name = key[prev_dot+1:weight_idx]
                break
    print(f'prefix is found {prefix}, lora_name is {lora_name}')

    if len(prefix) > 0:
        #replace the prefix to empty
        for key in keys_in_hf:
            if key.startswith(prefix):
                new_key = key[len(prefix)+1:]
                if len(lora_name) > 0:
                    new_key = new_key.replace(lora_name+".","")
                    if new_key.endswith(lora_name):
                        new_key = new_key[:len(new_key)-len(lora_name)-1]
                if new_key.startswith("emb."):
                    new_key = new_key.replace("lora_embedding_","lora_")
                else:
                    new_key = new_key[0:-len(".weight")]
                rwkv_lora_state_dict[new_key] = hf_lora_state_dict[key]

            else:
                rwkv_lora_state_dict[key] = hf_lora_state_dict[key]

    print(f'rwkv_lora_state_dict keys {rwkv_lora_state_dict.keys()}')

    if args.rwkv_lora is None:
        base_name = args.hf_lora.split(".")[0]
        rwkv_lora_file = f"{base_name}_rwkv_lora.pth"
        args.rwkv_lora = rwkv_lora_file
    torch.save(rwkv_lora_state_dict, args.rwkv_lora)