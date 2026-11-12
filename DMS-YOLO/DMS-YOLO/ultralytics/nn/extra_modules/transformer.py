

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2PSA, PSABlock
from .attention import AttentionTSSA
from .mona import Mona
from .semnet import SEFN


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.7):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class PSABlock_DYT(PSABlock):

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

    def forward(self, x):
        x = x + self.attn(self.dyt1(x)) if self.add else self.attn(self.dyt1(x))
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x


class DTABlock(C2PSA):

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(
            PSABlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64)
            for _ in range(n)
        ))


class MFE(nn.Module):

    def __init__(self, c, num_heads=4):
        super().__init__()
        self.attn = AttentionTSSA(c, num_heads=num_heads)
        self.mona = Mona(c)
        self.ffn = SEFN(c, ffn_expansion_factor=2, bias=False)

class DMSUnit(PSABlock):

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        del self.attn
        del self.ffn
        self.mfe = MFE(c, num_heads=num_heads)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

    def forward(self, x):
        x_spatial = x
        _, C, H, W = x.size()
        attn_tokens = self.dyt1(x).flatten(2).permute(0, 2, 1)
        attn_x = self.mfe.attn(attn_tokens).permute(0, 2, 1).view(-1, C, H, W).contiguous()
        x = x + attn_x if self.add else attn_x
        x = self.mfe.mona(x)
        ffn_x = self.mfe.ffn(self.dyt2(x), x_spatial)
        x = x + ffn_x if self.add else ffn_x
        return x


class DMSBlock(C2PSA):

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(
            DMSUnit(self.c, attn_ratio=0.5, num_heads=self.c // 64)
            for _ in range(n)
        ))
