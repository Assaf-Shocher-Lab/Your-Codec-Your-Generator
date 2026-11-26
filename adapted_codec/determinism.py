"""Exact float32 arithmetic and deterministic cuDNN: the decoder's search must be a deterministic function of the
received record."""
import torch


def set_exact_math():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
