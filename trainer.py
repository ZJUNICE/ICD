from collections import defaultdict
from dataclasses import dataclass
import time
import copy
from typing import Optional, Tuple
import numpy as np
import os

from simple_parsing.helpers import Serializable
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW
# from torch.optim import AdamW
from thop import profile

def calculate_flops_thop(model, inputs):
    """使用thop计算FLOPs并屏蔽所有输出"""
    import sys, os
    
    # 保存原始stdout
    original_stdout = sys.stdout
    # 重定向输出到/dev/null
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # 在静默模式下调用thop
        macs, params = profile(model, inputs=inputs)
        flops = macs * 2
    finally:
        # 恢复原始stdout
        sys.stdout.close()
        sys.stdout = original_stdout
    
    return flops, params


@dataclass
class TrainArgs(Serializable):
    device         : str                 = 'auto'
    num_workers    : int                 = 1
    max_iters      : Optional[int]       = None
    batch_size     : int                 = 1
    learning_rate  : float               = 3e-4
    betas          : Tuple[float, float] = (0.9, 0.95)
    weight_decay   : float               = 0.1
    grad_norm_clip : float               = 1.0


class Trainer(object):
    @staticmethod
    def get_default_config():
        C = TrainArgs()
        return C

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=float(config.learning_rate))
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        self.flops_sum = 0
        print("running on device", self.device)

    def update_step(self, batch):
        self.model.eval()  # only for inference
        
        inputs = batch['x']
        labels = batch['y']
        
        logits = self.predict_step(inputs)
        logits = logits.detach()
        # loss = self.loss_step(inputs, labels)
        # self.optim_step(loss)
        probs = F.softmax(logits[0][-1], dim=-1)

        return probs

    def predict_step(self, inputs):
        with torch.no_grad(): 
            inputs = torch.LongTensor([inputs]).to(self.device)
            # 计算FLOPs
            flops, params = calculate_flops_thop(self.model, inputs)
            self.flops_sum += flops / 1e9  # 转换为G
            outputs = self.model(input_ids=inputs)
            logits = outputs.logits
        return logits

    def loss_step(self, inputs, labels):
        inputs = torch.LongTensor([inputs]).to(self.device)
        labels = torch.LongTensor([labels]).to(self.device)
        
        outputs = self.model(
            input_ids=inputs,
            labels=labels
        )
        loss = outputs.loss

        return loss
    
    def optim_step(self, loss):
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_norm_clip
        )
        self.optimizer.step()

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}.")
