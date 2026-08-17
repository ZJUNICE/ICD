from munch import munchify

import argparse
import random
import numpy as np
import torch
import yaml
import os

import file_io
from trainer import TrainArgs, Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_global_config(path: str):
    with open(path, 'rb') as f:
        config = yaml.safe_load(f)
    config = munchify(config)
    return config


def init_by_config_path(
    input_path: str,
    output_path: str,
    config_path: str,
    mode: str,
):
    assert mode == 'compress' or mode == 'decompress', \
        f'Invalid mode {mode}. Choose either \'compress\' or \'decompress\'.'
    config = get_global_config(config_path)
    set_seed(config.seed)

    # model_name = config.model_name  # online loading
    model_path = config.model_path  # local loading


    # Tokenizer
    from transformers import GPT2Tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)


    # Model
    from transformers import GPT2LMHeadModel
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    config.block_size = model.config.n_ctx   # maximum input sequence length of model
    config.vocab_size = len(tokenizer)       # size of vocabulary of model
    config.bos_idx = tokenizer.bos_token_id
    config.eos_idx = tokenizer.eos_token_id
    config.unk_idx = tokenizer.unk_token_id
    
    # Trainer
    trainer_args = TrainArgs.from_dict(config.trainer)
    trainer = Trainer(trainer_args, model)


    # io
    io_mode = ''
    io_cls = getattr(file_io, config.ac.io)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if mode == 'compress':
        io_mode += 'a'
        if 'Binary' in config.ac.io: io_mode += 'b'
        io = io_cls(output_path, io_mode)
    else:
        io_mode += 'r'
        if 'Binary' in config.ac.io: io_mode += 'b'
        io = io_cls(input_path, io_mode)


    return config, tokenizer, trainer, io


def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def SNR_to_std(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()