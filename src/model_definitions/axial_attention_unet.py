import torch
from torch import nn
from axial_attention import AxialAttention

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, 3, padding='same')
        self.conv2 = nn.Conv2d(out_features, out_features, 3, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class AttentionBlock1(nn.Module):
    def __init__(
        self, dim, dim_index=1, dim_heads=32, heads=8, num_dimensions=2,
        sum_axial_out=True
        ):
        super().__init__()
        
        self.attn = AxialAttention(
            dim=dim, dim_index=dim_index, dim_heads=dim_heads, 
            num_dimensions=num_dimensions, sum_axial_out=sum_axial_out
            )
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding='same')
        self.conv1 = nn.Conv2d(dim, 2*dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(dim, 2*dim, kernel_size=3, stride=1, padding='same')
        self.pool = nn.AvgPool2d(2,2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.relu(self.conv(x))
        x2 = self.attn(x2)
        x2 = self.relu(self.pool(x2))
        x2 = self.conv2(x2)
        
        x = torch.add(x1, x2)
        x = self.relu(x)

        return x


class AttentionBlock2(nn.Module):
    def __init__(
        self, dim, dim_index=1, dim_heads=32, heads=8, num_dimensions=2, 
        sum_axial_out=True
        ):
        super().__init__()

        self.attn = AxialAttention(
            dim=dim, dim_index=dim_index, dim_heads=dim_heads, 
            num_dimensions=num_dimensions, sum_axial_out=sum_axial_out
            )
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)

        x2 = self.relu(self.conv(x))
        x2 = self.attn(x2)
        x2 = self.relu(x2)
        x2 = self.conv(x2)
        
        x = torch.add(x1, x2)
        x = self.relu(x)

        return x


class AxialAttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn1 = AttentionBlock1(dim)
        self.attn2 = AttentionBlock2(dim*2)

    def forward(self, x):
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.attn2(x)
        x = self.attn2(x)

        return x


class model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Contracting path convolutions
        self.conv64 = ConvBlock(1, 64)
        self.conv128 = ConvBlock(64, 128)
        self.conv256 = ConvBlock(128, 256)
        self.attn512 = AxialAttentionLayer(256)
        self.attn1024 = AxialAttentionLayer(512)
        
        self.pool = nn.MaxPool2d(2,2)
        
        # Expanding path convolutions
        self.convup512 = ConvBlock(1024, 512)
        self.convup256 = ConvBlock(512, 256)
        self.convup128 = ConvBlock(256, 128)
        self.convup64 = ConvBlock(128, 64)

        self.convtran512 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.convtran256 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.convtran128 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.convtran64 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.head = nn.Conv2d(64, 1, 1, padding='same')
        
        self.relu = nn.ReLU()


    def forward(self, x):
        # Contracting path
        x64 = self.conv64(x)
        pool_64 = self.pool(x64)
        x128 = self.conv128(pool_64)
        pool_128 = self.pool(x128)
        x256 = self.conv256(pool_128)
        x512 = self.attn512(x256)
        x1024 = self.attn1024(x512)

        # Expanding path
        xup512 = self.convtran512(x1024)
        xup512 = torch.cat((xup512,x512), axis=1)
        xup512 = self.convup512(xup512)

        xup256 = self.convtran256(xup512)
        xup256 = torch.cat((xup256,x256), axis=1)
        xup256 = self.convup256(xup256)
        
        xup128 = self.convtran128(xup256)
        xup128 = torch.cat((xup128,x128), axis=1)
        xup128 = self.convup128(xup128)
        
        xup64 = self.convtran64(xup128)
        xup64 = torch.cat((xup64,x64), axis=1)
        xup64 = self.convup64(xup64)

        # Output
        out = self.head(xup64)

        return xup64