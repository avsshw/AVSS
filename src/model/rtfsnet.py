import torch
import torch.nn as nn
from einops import rearrange


class STFT(nn.Module):
    """Short-Time Fourier Transform encoder"""

    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x):
        """
        Args:
            x: (batch, time) waveform
        Returns:
            stft: (batch, freq, time, 2) complex STFT
        """
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        stft_real = torch.stack([stft.real, stft.imag], dim=-1)
        return stft_real


class ISTFT(nn.Module):
    """Inverse Short-Time Fourier Transform decoder"""

    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, stft_real, length=None):
        """
        Args:
            stft_real: (batch, freq, time, 2) complex STFT representation
            length: target output length
        Returns:
            x: (batch, time) waveform
        """
        stft_complex = torch.complex(stft_real[..., 0], stft_real[..., 1])

        x = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=length,
        )
        return x


class FrequencyRNN(nn.Module):
    """RNN processing along frequency dimension"""

    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        """
        Args:
            x: (batch, freq, time, features)
        Returns:
            out: (batch, freq, time, hidden_size*2)
        """
        batch, freq, time, features = x.shape
        x = rearrange(x, "b f t d -> (b t) f d")

        out, _ = self.rnn(x)

        out = rearrange(out, "(b t) f d -> b f t d", b=batch)

        out = self.layer_norm(out)

        return out


class TimeRNN(nn.Module):
    """RNN processing along time dimension"""

    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        """
        Args:
            x: (batch, freq, time, features)
        Returns:
            out: (batch, freq, time, hidden_size*2)
        """
        batch, freq, time, features = x.shape
        x = rearrange(x, "b f t d -> (b f) t d")

        out, _ = self.rnn(x)

        out = rearrange(out, "(b f) t d -> b f t d", b=batch)

        out = self.layer_norm(out)

        return out


class TFInteraction(nn.Module):
    """Efficient Time-Frequency Interaction Layer"""

    def __init__(self, embed_dim):
        super().__init__()
        self.freq_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.time_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, freq, time, features)
        Returns:
            out: (batch, freq, time, features)
        """
        batch, freq, time, features = x.shape
        residual = x

        x_freq = rearrange(x, "b f t d -> (b t) d f")
        x_freq = self.freq_conv(x_freq)
        x_freq = rearrange(x_freq, "(b t) d f -> b f t d", b=batch)

        x_time = rearrange(x, "b f t d -> (b f) d t")
        x_time = self.time_conv(x_time)
        x_time = rearrange(x_time, "(b f) d t -> b f t d", b=batch)

        x = self.layer_norm(residual + x_freq + x_time)

        ffn_out = self.ffn(x)
        out = self.layer_norm2(x + ffn_out)

        return out


class RTFSBlock(nn.Module):
    """
    Recurrent Time-Frequency Separation Block
    Processes frequency dimension, then time dimension, then applies TF interaction
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.freq_rnn = FrequencyRNN(hidden_size, hidden_size, num_layers)

        self.time_rnn = TimeRNN(hidden_size * 2, hidden_size, num_layers)

        self.tf_interaction = TFInteraction(hidden_size * 2)

        self.output_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        """
        Args:
            x: (batch, freq, time, input_size)
        Returns:
            out: (batch, freq, time, hidden_size)
        """
        x = self.input_proj(x)
        residual = x

        x = self.freq_rnn(x)

        x = self.time_rnn(x)

        x = self.tf_interaction(x)

        out = self.output_proj(x) + residual

        return out


class RTFSNet(nn.Module):
    """
    RTFS-Net: Recurrent Time-Frequency Separation Network
    Audio-only version for source separation
    """

    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        win_length=512,
        num_blocks=4,
        hidden_size=128,
        rnn_layers=2,
        num_sources=2,
    ):
        """
        Args:
            n_fft: FFT size
            hop_length: hop length for STFT
            win_length: window length for STFT
            num_blocks: number of RTFS blocks
            hidden_size: hidden size for RNNs
            rnn_layers: number of RNN layers
            num_sources: number of sources to separate
        """
        super().__init__()

        self.num_sources = num_sources
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.encoder = STFT(n_fft, hop_length, win_length)

        self.decoder = ISTFT(n_fft, hop_length, win_length)

        input_size = 2

        self.blocks = nn.ModuleList(
            [
                RTFSBlock(
                    input_size if i == 0 else hidden_size, hidden_size, rnn_layers
                )
                for i in range(num_blocks)
            ]
        )

        self.mask_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_sources * 2),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)
                        hidden_size = m.hidden_size
                        param.data[hidden_size : 2 * hidden_size].fill_(1.0)

    def forward(self, mix):
        """
        Args:
            mix: (batch, time) mixed waveform
        Returns:
            sources: (batch, num_sources, time) separated sources
        """
        original_length = mix.shape[-1]

        spec = self.encoder(mix)
        batch, freq, time, _ = spec.shape

        x = spec
        for block in self.blocks:
            x = block(x)

        masks = self.mask_estimator(x)
        masks = rearrange(masks, "b f t (s c) -> b f t s c", s=self.num_sources, c=2)

        spec_expanded = rearrange(spec, "b f t c -> b f t 1 c")

        real_part = (
            spec_expanded[..., 0] * masks[..., 0]
            - spec_expanded[..., 1] * masks[..., 1]
        )
        imag_part = (
            spec_expanded[..., 0] * masks[..., 1]
            + spec_expanded[..., 1] * masks[..., 0]
        )

        masked_specs = torch.stack([real_part, imag_part], dim=-1)

        sources = []
        for i in range(self.num_sources):
            source_spec = masked_specs[:, :, :, i, :]
            source_wav = self.decoder(source_spec, length=original_length)
            sources.append(source_wav)

        sources = torch.stack(sources, dim=1)

        return sources
