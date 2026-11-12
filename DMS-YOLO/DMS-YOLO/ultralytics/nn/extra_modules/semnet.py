
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type != 'WithBias':
            raise ValueError('DMS-YOLO only uses WithBias LayerNorm')
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class SEFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SEFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)


        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
        )
        self.upsample = nn.Upsample(scale_factor=2)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(4, dim // 32), 1, bias=False),  # 极小的中间层
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, dim // 32), dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, spatial):
        identity = x

        ca_weights = self.channel_attention(x)
        enhanced_x = x * ca_weights

        x = self.project_in(enhanced_x)
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)

        if y.shape[-2:] != spatial.shape[-2:]:
            y = F.interpolate(y, size=spatial.shape[-2:], mode="nearest")

        y_ca_weights = self.channel_attention(y)
        y = y * y_ca_weights

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))

        dwconv_weight_slice = self.dwconv.weight[:x1.size(1)]
        x1 = F.conv2d(x1, dwconv_weight_slice, padding=1, groups=x1.size(1))

        x = F.relu(x1) * x2
        x = self.project_out(x)

        return 0.7 * x + 0.3 * identity
