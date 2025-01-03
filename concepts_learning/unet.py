import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Double Convolution block:
    (Conv2d -> BatchNorm -> ReLU) × 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling block:
    MaxPool2d -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling block:
    ConvTranspose2d -> Concatenate -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, 
            kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handling input sizes that are not perfectly divisible by 2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2
        ])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        Args:
            in_channels (int): number of input channels (e.g., 3 for RGB)
            out_channels (int): number of output channels
            features (list): feature dimensions for each layer
        """
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        in_features = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_features, feature))
            in_features = feature

        # Bottom part
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                Up(feature*2, feature)
            )

        # Final convolution
        self.final_conv = nn.Conv2d(
            features[0], out_channels, kernel_size=1
        )

    def forward(self, x):
        skip_connections = []

        # Down path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse list

        # Up path
        for idx in range(len(self.ups)):
            x = self.ups[idx](x, skip_connections[idx])

        return self.final_conv(x)

# Example usage:
def test_unet():
    x = torch.randn((3, 3, 161, 161))  # (batch_size, channels, height, width)
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    # Calculate total parameters
    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total trainable parameters: {total_params:,}")

if __name__ == "__main__":
    test_unet()
