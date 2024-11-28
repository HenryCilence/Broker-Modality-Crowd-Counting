from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from timm.layers import DropPath


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x



class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()

        c = [32, 64, 128, 256]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        # cross attention
        self.cross_att = CrossAttentionTransformer(embed_dim=256, depth=1, num_heads=4)
        
        self.pre_conv_rgb = conv_block(in_ch, in_ch)
        self.pre_conv_t = conv_block(in_ch, in_ch)

        self.conv1 = conv_block(in_ch * 2, c[0])
        self.conv2 = conv_block(c[0], c[1])
        self.conv3 = conv_block(c[1], c[2])
        self.conv4 = conv_block(c[2], c[3])

        self.Up4 = up_conv(c[3], c[2])
        self.Up_conv4 = conv_block(c[3], c[2])

        self.Up3 = up_conv(c[2], c[1])
        self.Up_conv3 = conv_block(c[2], c[1])

        self.Up2 = up_conv(c[1], c[0])
        self.Up_conv2 = conv_block(c[1], c[0])

        self.conv = nn.Conv2d(c[0], out_ch, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, t):
        cross_att = self.cross_att(rgb, t)
        rgb = self.pre_conv_rgb(rgb)
        t = self.pre_conv_rgb(t)
        x = torch.cat((rgb, t), dim=1)

        e1 = self.conv1(x)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)  # c=c[3]
        e4 = e4 + cross_att

        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1) 
        d4 = self.Up_conv4(d4)  # c=c[2]

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)  # c=c[1]

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)  # c=c[0]

        out = self.conv(d2)

        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.1,
                 proj_drop_ratio=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x_q, x_kv):
        assert x_q.shape == x_kv.shape
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x_q.shape

        # kv(): -> [batch_size, num_patches + 1, 2 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 2, num_heads, embed_dim_per_head]
        # permute: -> [2, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        kv = self.kv(x_kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = self.q(x_q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.1,
                 attn_drop_ratio=0.1,
                 drop_path_ratio=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(CrossAttentionBlock, self).__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x[0] + self.drop_path1(self.attn(self.norm1_q(x[0]), self.norm1_kv(x[1])))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class CrossAttentionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=512, num_classes=1000,
                 embed_dim=256, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.1,
                 attn_drop_ratio=0.1, drop_path_ratio=0.1, norm_layer=None,
                 act_layer=None):
        super(CrossAttentionTransformer, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm = norm_layer(embed_dim)
        self.feature = self.make_layers(self.cfg['A'])

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            CrossAttentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M'],
    }

    def forward(self, rgb, t):
        assert rgb.shape == t.shape
        B, C, H, W = rgb.shape

        feature_rgb = self.feature(rgb)
        feature_t = self.feature(t)
        assert feature_rgb.shape == feature_t.shape
        B, C, H, W = feature_rgb.shape

        feature_rgb = feature_rgb.flatten(2).transpose(1, 2)
        feature_t = feature_t.flatten(2).transpose(1, 2)

        x = self.blocks((feature_rgb, feature_t))
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x
