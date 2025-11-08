import math
import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        use_pretrained: bool = True,
        pretrained_path: str | None = None,
        flat_input_dim: int | None = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.flat_input_dim = flat_input_dim
        self.pretrained_loaded = False

        # Conv pathway for per-frame images (grayscale mouth crops)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, feature_dim)

        # Optional MLP pathway for flat per-frame feature vectors (size must be known)
        self.mlp: nn.Module | None = None
        if self.flat_input_dim is not None:
            self.mlp = nn.Sequential(
                nn.Linear(self.flat_input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, feature_dim),
            )

        if use_pretrained and pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, pretrained_path: str):
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state = checkpoint.get("state_dict", checkpoint)
            own_state = self.state_dict()

            # Greedy by-shape loading to salvage as much as possible
            used = set()
            assign = {}
            for name, param in own_state.items():
                target_shape = param.shape
                # exact name match first
                if name in state and state[name].shape == target_shape:
                    assign[name] = state[name]
                    used.add(name)
                    continue
                # otherwise, search by shape
                for k, v in state.items():
                    if k in used:
                        continue
                    if isinstance(v, torch.Tensor) and v.shape == target_shape:
                        assign[name] = v
                        used.add(k)
                        break
            missing, unexpected = self.load_state_dict(assign, strict=False)
            # Freeze loaded layers
            for p in self.parameters():
                p.requires_grad = False
            self.pretrained_loaded = True
        except Exception:
            # Fallback: leave randomly initialized, trainable
            self.pretrained_loaded = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
            video:
              - images: (batch, num_speakers, t, h, w)
              - flat:   (batch, num_speakers, t, d) where d is known (set flat_input_dim in config)
        Returns:
            features: (batch, num_speakers, t, feature_dim)
        """
        if video.dim() == 5:
            # (b, s, t, h, w) -> (b*s*t, 1, h, w)
            b, s, t, h, w = video.shape
            frames = video.view(b * s * t, 1, h, w)
            x = self.conv(frames).view(b * s * t, -1)  # (b*s*t, 64)
            x = self.fc(x)  # (b*s*t, feature_dim)
            return x.view(b, s, t, self.feature_dim)

        if video.dim() == 4:
            b, s, t, d = video.shape
            if self.mlp is not None:
                x = video.view(b * s * t, d)
                x = self.mlp(x)
                return x.view(b, s, t, self.feature_dim)
            # try to interpret as flattened square image
            h = int(math.isqrt(d))
            if h * h == d:
                frames = video.view(b * s * t, 1, h, h)
                x = self.conv(frames).view(b * s * t, -1)
                x = self.fc(x)
                return x.view(b, s, t, self.feature_dim)
            raise ValueError(
                "Flat video features detected with unknown dimension. "
                "Set config.model.video_flat_input_dim to the per-frame feature size."
            )

        raise ValueError("Unsupported video tensor shape")
