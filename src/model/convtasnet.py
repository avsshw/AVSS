import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, L=20, N=64):
        super().__init__()
        self.conv = nn.Conv1d(1, N, L, stride=L // 2, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.relu(self.conv(x))


class Decoder(nn.Module):
    def __init__(self, L=20, N=64):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(N, 1, L, stride=L // 2, bias=False)

    def forward(self, x):
        return self.deconv(x).squeeze(1)


class Separator(nn.Module):
    def __init__(self, N=64, num_sources=2):
        super().__init__()
        self.num_sources = num_sources
        self.N = N
        self.mask_gen = nn.Linear(N, N * num_sources)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.mask_gen(x)
        x = x.view(x.shape[0], x.shape[1], self.num_sources, self.N)
        x = x.permute(0, 2, 3, 1)
        return F.relu(x)


class SimpleConvTasNet(nn.Module):
    def __init__(self, L=20, N=64, num_sources=2):
        super().__init__()
        self.encoder = Encoder(L, N)
        self.separator = Separator(N, num_sources)
        self.decoder = Decoder(L, N)
        self.num_sources = num_sources
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, mix):
        encoded = self.encoder(mix)
        masks = self.separator(encoded)

        outputs = []
        for i in range(self.num_sources):
            masked = encoded * masks[:, i]
            outputs.append(self.decoder(masked))

        return torch.stack(outputs, dim=1)
