"""
Chapter 7: Neural Operators
1D Fourier Neural Operator from scratch
"""

import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.weights = nn.Parameter(
            torch.empty(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        nn.init.xavier_normal_(self.weights)
        
    def forward(self, x):
        # x: [batch, in_channels, x_resolution]
        batch_size = x.shape[0]
        
        # Forward FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1)//2 + 1, 
                             device=x.device, dtype=torch.cfloat)
        
        # Complex multiplication using einsum
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :self.modes], 
            self.weights
        )
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64):
        super().__init__()
        self.p = nn.Linear(2, width) # input + spatial location x
        
        self.conv1 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, 1)
        
        self.conv2 = SpectralConv1d(width, width, modes)
        self.w2 = nn.Conv1d(width, width, 1)
        
        self.q = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.act = nn.GELU()
        
    def forward(self, x):
        # x: [batch, grid_size, 2]
        x = self.p(x)
        x = x.permute(0, 2, 1) # [batch, width, grid_size]
        
        x1 = self.conv1(x) + self.w1(x)
        x1 = self.act(x1)
        
        x2 = self.conv2(x1) + self.w2(x1)
        x2 = self.act(x2)
        
        x2 = x2.permute(0, 2, 1) # [batch, grid_size, width]
        x_out = self.q(x2)
        return x_out

def main():
    print("Initializing 1D Fourier Neural Operator...")
    model = FNO1d(modes=16, width=32)
    
    # Dummy input: [batch, resolution, channels(u(x), x)]
    batch_size = 10
    resolution = 128
    dummy_input = torch.randn(batch_size, resolution, 2)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("FNO1d forward pass successful.")

if __name__ == "__main__":
    main()
