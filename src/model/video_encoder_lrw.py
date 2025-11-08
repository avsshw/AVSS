import torch
import torch.nn as nn
from typing import Optional


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (b, 512, h', w')


class VideoEncoderLRW(nn.Module):
    """
    ResNet-18 visual frontend (grayscale mouth crops) producing 512-D per-frame embeddings.
    Loads weights greedily from LRW video checkpoints and freezes loaded params.
    """

    def __init__(self, pretrained_path: Optional[str] = None, freeze_backbone: bool = True, in_channels: int = 1):
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels=in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path, freeze=freeze_backbone)

    @torch.no_grad()
    def _load_pretrained(self, path: str, freeze: bool = True):
        try:
            state = torch.load(path, map_location="cpu")
            state = state.get("state_dict", state)
            own = self.backbone.state_dict()
            assign = {}
            used = set()
            for name, param in own.items():
                if name in state and isinstance(state[name], torch.Tensor) and state[name].shape == param.shape:
                    assign[name] = state[name]
                    used.add(name)
                    continue
                # fallback: search by shape
                for k, v in state.items():
                    if k in used:
                        continue
                    if isinstance(v, torch.Tensor) and v.shape == param.shape:
                        assign[name] = v
                        used.add(k)
                        break
            self.backbone.load_state_dict(assign, strict=False)
            if freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False
        except Exception:
            pass

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (b, num_speakers, t, h, w) grayscale mouth crops
        Returns:
            feats: (b, num_speakers, t, 512)
        """
        b, s, t, h, w = video.shape
        x = video.view(b * s * t, 1, h, w)
        x = self.backbone(x)               # (b*s*t, 512, h', w')
        x = self.pool(x).view(b * s * t, 512)
        x = x.view(b, s, t, 512)
        return x


