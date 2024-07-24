if __name__ == '__main__':
    import torch
    import argparse
    parser = argparse.ArgumentParser("Extract encoder from MAE model")
    parser.add_argument("--input_model_path", type=str, required=True, help="Path to the MAE model")
    parser.add_argument("--output_encoder_path", type=str, required=True, help="Path to save the encoder")
    args = parser.parse_args()
    original_weights = torch.load(args.input_model_path)
    new_weights = {}
    for key in original_weights:
        if not 'onelayer_decoder' in key:
            new_weights[key] = original_weights[key]
    torch.save(new_weights, args.output_encoder_path)
    print('Encoder extracted and saved to', args.output_encoder_path)
    print(original_weights.keys())
    print('->')
    print(new_weights.keys())