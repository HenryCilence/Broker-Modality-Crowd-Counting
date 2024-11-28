from collections import OrderedDict
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from models.vit import VisionTransformer
from models.unet_cross_attention import U_Net

unet_path = r"/home/home/menghaoliang/code/count/distillation/check/unet_cross_att/1227-184622/unet_cross_att.pth"


class BM(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = vgg19()
        self.unet = unet()
        self.encoder = encoder()
        self.reg_layer_0 = reg()
        print(self.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'), strict=False))

    def forward(self, inputs):
        rgb, t = inputs
        rgbt = self.unet(rgb, t)
        rgb_feature_1 = self.features(rgb)
        t_feature_1 = self.features(t)
        d_feature_1 = self.features(rgbt)

        rgb_feature_2 = self.encoder(rgb_feature_1)
        t_feature_2 = self.encoder(t_feature_1)
        d_feature_2 = self.encoder(d_feature_1)

        rgb_feature_2 = F.interpolate(rgb_feature_2, scale_factor=2)
        t_feature_2 = F.interpolate(t_feature_2, scale_factor=2)
        d_feature_2 = F.interpolate(d_feature_2, scale_factor=2)

        fusion = rgb_feature_2 + t_feature_2 + d_feature_2
        fusion /= 3

        density = self.reg_layer_0(fusion)

        return torch.abs(density)


def vgg19():
    model = make_layers((cfg["E"]))
    return model


def reg():
    model = nn.Sequential(
        nn.Conv2d(768, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 1, 1)
    )
    return model


def encoder():
    return VisionTransformer(embed_dim=768, depth=2, num_heads=6)


def unet():
    model = U_Net()
    # model.load_state_dict(torch.load(unet_path))
    return model

def make_layers(cfg, batch_norm=False):
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
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


if __name__ == "__main__":
    model = BM()
    rgb = torch.randn((1, 3, 224, 224))
    t = torch.randn((1, 3, 224, 224))
    output = model([rgb, t])
    print(output.size())
