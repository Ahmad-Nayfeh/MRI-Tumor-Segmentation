# src/models.py (Final Corrected Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ======================================================================================
# Building Block: Double Convolutional Block
# ======================================================================================

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ======================================================================================
# Model 1: Baseline U-Net
# ======================================================================================

class BaselineUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=[64, 128, 256, 512]):
        super(BaselineUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder (Upsampling path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse for decoder

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return torch.sigmoid(self.final_conv(x))

# ======================================================================================
# Model 2: U-Net with a Pre-trained ResNet34 Backbone
# ======================================================================================

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(ResNetUNet, self).__init__()

        self.base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Modify the first convolution layer to accept 4 channels
        original_weights = self.base_model.conv1.weight.data
        new_first_layer_weights = torch.cat([original_weights, original_weights[:, 0:1, :, :]], dim=1)
        
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.conv1.weight.data = new_first_layer_weights
        
        self.base_layers = list(self.base_model.children())

        # Encoder layers
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # -> 64 channels
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # -> 64 channels
        self.layer2 = self.base_layers[5] # -> 128 channels
        self.layer3 = self.base_layers[6] # -> 256 channels
        self.layer4 = self.base_layers[7] # -> 512 channels

        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv4 = DoubleConv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv1 = DoubleConv(128, 64)

        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder
        u4 = self.upconv4(x4)
        
        # *** THE FIX IS HERE ***
        # Resize u4 to match x3's spatial dimensions before concatenation
        if u4.shape[2:] != x3.shape[2:]:
            u4 = F.interpolate(u4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec_conv4(torch.cat([u4, x3], dim=1))

        u3 = self.upconv3(d4)
        if u3.shape[2:] != x2.shape[2:]:
            u3 = F.interpolate(u3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec_conv3(torch.cat([u3, x2], dim=1))

        u2 = self.upconv2(d3)
        if u2.shape[2:] != x1.shape[2:]:
            u2 = F.interpolate(u2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec_conv2(torch.cat([u2, x1], dim=1))

        u1 = self.upconv1(d2)
        if u1.shape[2:] != x0.shape[2:]:
            u1 = F.interpolate(u1, size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec_conv1(torch.cat([u1, x0], dim=1))

        u0 = self.upconv0(d1)
        
        # Final resize to match original input size, just in case
        if u0.shape[2:] != x.shape[2:]:
            u0 = F.interpolate(u0, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        return torch.sigmoid(self.final_conv(u0))

# ======================================================================================
# Model 3: TransUNet (Simplified, Runnable Version)
# ======================================================================================

class TransUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(TransUNet, self).__init__()
        
        self.cnn_encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            DoubleConv(32, 32),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        cnn_features = self.cnn_encoder(x)
        output = self.decoder(cnn_features)
        
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        return torch.sigmoid(output)


# ======================================================================================
# Sanity Check: Test models with a dummy input
# ======================================================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Testing Model Architectures on device: {device} ---")
    
    dummy_input = torch.randn(2, 4, 240, 240).to(device)
    
    def test_model(model_class, model_name):
        print(f"\n[{test_model.counter}] Testing {model_name}...")
        test_model.counter += 1
        model = model_class(in_channels=4, out_channels=1).to(device)
        output = model(dummy_input)
        
        expected_shape = (dummy_input.shape[0], 1, dummy_input.shape[2], dummy_input.shape[3])
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == expected_shape, f"{model_name} output shape is incorrect!"
        print(f"  {model_name} test PASSED!")

    test_model.counter = 1
    
    test_model(BaselineUNet, "BaselineUNet")
    test_model(ResNetUNet, "ResNetUNet")
    test_model(TransUNet, "TransUNet")
    
    print("\n--- All model tests completed successfully! ---")