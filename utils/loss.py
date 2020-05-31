import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from config import Config

cfg = Config()


class LatexLoss(nn.Module):
    def __init__(self):
        super(LatexLoss, self).__init__()

    def forward(self, logits, tokens):
        # logits: shape of [N, MAX_FORMULA_LENGTH, TOKEN_COUNT], value is [0,1]
        # tokens: shape of [N, MAX_FORMULA_LENGTH] value is index of token
        pos_logits = torch.gather(logits, dim=1, index=tokens)
        loss = -1 * torch.log(pos_logits)
        return loss.sum() / cfg.BATCH_SIZE








