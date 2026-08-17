import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import data_loader
from utils import init_by_config_path, set_seed
import conf

def compress(sentence, config, tokenizer, trainer, io, count, min_width):
    # setup global variables for compressing
    precision = config.ac.precision
    top_value = 2 ** precision
    quarter = top_value // 4
    half = 2 * quarter
    
    token = tokenizer.tokenize(sentence)  # BPE tokenization
    print(f'\nToken length of current sentence: {len(token)}')
    tokenID = tokenizer.encode(token)
    
    data_loader_cls = getattr(data_loader, config.dataloader) 
    dataloader = data_loader_cls(tokenID, tokenizer, config.block_size)

    low = 0
    high = top_value
    bits_to_follow = 0
    
    for batch in dataloader:
        count += 1
        width = high - low + 1
        if width == 1 or width == 0:
            print('Precision error when compressing.')
            exit(-1)

        probs = trainer.update_step(batch)

        probs = probs.to(torch.float64)

        tgt_idx = batch['y'][-1]
        # print(f'tgt_idx = {tgt_idx}')
        
        cumprobs = torch.cumsum(probs, dim=0)
        cumprobs = torch.cat(
            (torch.tensor([0.0], device=probs.device), cumprobs), dim=0
        )
        cumprobs = cumprobs.to(torch.float64)
        cumprob_high = cumprobs[tgt_idx + 1].to('cpu').item()
        cumprob_low = cumprobs[tgt_idx].to('cpu').item()

        high = low + int(width * cumprob_high)
        low = low + int(width * cumprob_low)
        
        prob_item = probs[tgt_idx].to('cpu').item()
        width_new = width * probs[tgt_idx].to('cpu').item()
        assert torch.isclose(torch.tensor(prob_item), torch.tensor(cumprob_high - cumprob_low), \
                            atol=1e-16), "Insufficient float precision for 'probs' and 'cumprobs'."
        min_width = width_new if (width_new < min_width) else min_width
        assert low != high, "Width of subinterval less than 1."

        while high < half or low >= half:
            if high < half:
                io.write([0] + bits_to_follow*[1])
                bits_to_follow = 0
                low = 2 * low
                high = 2 * high
            elif low >= half:
                io.write([1] + bits_to_follow*[0])
                bits_to_follow = 0
                low = 2 * (low - half)
                high = 2 * (high - half)
        while low >= quarter and high < 3 * quarter:
            low = 2 * (low - quarter)
            high = 2 * (high - quarter)
            bits_to_follow += 1

    bits_to_follow += 1
    if low < quarter:
        io.write([0] + bits_to_follow*[1])
    else:
        io.write([1] + bits_to_follow*[0])

    return count, min_width


def main():
    set_seed(42)
    config, tokenizer, trainer, io = init_by_config_path(
        args.input_file, args.output_file, args.config_file, 'compress'
    )
    print('Start compressing with the following config...')
    print(config)
    
    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params/1e6:.4f}M")
    
    count = 0  # count of compressed tokens
    min_width = 2 ** config.ac.precision  # initial width of precision
    
    
    with open(args.input_file, 'r') as f:
        sentences = f.read().splitlines()  # split into sentence lists by line
    
    
    for i, sentence in enumerate(tqdm(sentences)):
        count, min_width = compress(sentence, config, tokenizer,
                                    trainer, io, count, min_width)
        if i < len(sentences) - 1:
            io.write(['\n'])
    
    
    io.close()
    
    print(f"\nMinimum width of subinterval: {min_width}")
    print(f'Compressed tokens count: {count}')
    print(f'Compressed to file {args.output_file}.')
    print(f'FLOPs: {trainer.flops_sum:.4f}G') #, Parameters: {params/1e6:.2f}M')



# global variables for compressing
parser = argparse.ArgumentParser(description=
                                 'Compress a text file while training an LLM.')
parser.add_argument('--input_file', type=str, 
                    default='mydata/demo.txt')
parser.add_argument('--output_file', type=str, 
                    default='mydata/demo_encode_out.txt')
parser.add_argument('--config_file', type=str, 
                    default='config/global/demo.yaml')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    main()