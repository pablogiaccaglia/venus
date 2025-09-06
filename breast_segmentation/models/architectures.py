"""Model architectures for breast segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import segmentation_models_pytorch as smp
from monai.networks.nets import UNet as MonaiUNet, SwinUNETR, BasicUNetPlusPlus
from monai.networks.layers import Norm


class DoubleConv(nn.Module):
    """Double convolutional block (conv -> BN -> ReLU) x2."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net architecture."""
    
    def __init__(self, n_channels: int = 1, n_classes: int = 1, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class NeighConv(nn.Module):
    """Neighborhood convolution for FCN-FFNET."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,  kernel_size=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FcnnFnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FcnnFnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.fc1 = OutConv(256, n_classes)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.fc2 = OutConv(128, n_classes)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.fc3 = OutConv(64, n_classes)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.neigh = NeighConv(n_classes, n_classes)

    def forward(self, x):

        activations = []
        out_fc = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        out1 = self.fc1(x)
        out_fc.append(out1)
        activations.append(x)
        x = self.up2(x, x3)
        out2 = self.fc2(x)
        out_fc.append(out2)
        activations.append(x)
        x = self.up3(x, x2)
        out3 = self.fc3(x)
        out_fc.append(out3)
        activations.append(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        neigh = self.neigh(logits)[:,:,:-1, :-1]
        return logits, out_fc, neigh, activations

class ConvBlock(nn.Module):
    """Convolutional block for SegNet."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 padding: int = 1, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block for SegNet."""
    
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        layers = []
        layers.append(ConvBlock(in_channels, out_channels))
        for _ in range(num_convs - 1):
            layers.append(ConvBlock(out_channels, out_channels))
        self.encode = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encode(x)
        x_pooled, indices = self.pool(x)
        return x_pooled, indices


class DecoderBlock(nn.Module):
    """Decoder block for SegNet."""
    
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        layers = []
        layers.append(ConvBlock(in_channels, in_channels))
        for _ in range(num_convs - 2):
            layers.append(ConvBlock(in_channels, in_channels))
        layers.append(ConvBlock(in_channels, out_channels))
        self.decode = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, indices: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
        x = self.unpool(x, indices, output_size=output_size)
        x = self.decode(x)
        return x


class SegNet(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)


        # Stage 5d
        torch.use_deterministic_algorithms(False)
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        torch.use_deterministic_algorithms(True)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        torch.use_deterministic_algorithms(False)
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        torch.use_deterministic_algorithms(True)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        torch.use_deterministic_algorithms(False)
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        torch.use_deterministic_algorithms(True)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        torch.use_deterministic_algorithms(False)
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        torch.use_deterministic_algorithms(True)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        torch.use_deterministic_algorithms(False)
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        torch.use_deterministic_algorithms(True)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)


def get_filters_count(level: int, base_filters: int) -> int:
    """Calculate the number of filters at each level."""
    return base_filters * (2 ** (level - 1))


class InceptionModule(nn.Module):
    """Inception module with 4 parallel branches."""
    def __init__(self, in_channels: int, out_channels: int,
                 activation=nn.LeakyReLU(0.3, inplace=True)):
        super().__init__()
        self.activation = activation
        branch_ch = out_channels // 4

        # Branch 1: 1×1
        self.branch1_1x1 = nn.Conv2d(in_channels, branch_ch, kernel_size=1)

        # Branch 2: 1×1 -> 3×3
        self.branch2_1x1 = nn.Conv2d(in_channels, branch_ch, kernel_size=1)
        self.branch2_3x3 = nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1)

        # Branch 3: 1×1 -> 5×5
        self.branch3_1x1 = nn.Conv2d(in_channels, branch_ch, kernel_size=1)
        self.branch3_5x5 = nn.Conv2d(branch_ch, branch_ch, kernel_size=5, padding=2)

        # Branch 4: MaxPool -> 1×1
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_1x1 = nn.Conv2d(in_channels, branch_ch, kernel_size=1)

    def forward(self, x):
        b1 = self.activation(self.branch1_1x1(x))
        b2 = self.activation(self.branch2_3x3(self.branch2_1x1(x)))
        b3 = self.activation(self.branch3_5x5(self.branch3_1x1(x)))
        b4 = self.activation(self.branch4_1x1(self.pool(x)))
        return torch.cat([b1, b2, b3, b4], dim=1)



class SkinnyNet(nn.Module):
    """
    'Skinny' U-Net
    """
    def __init__(self, image_channels=3, levels=6, base_filters=19):
        super().__init__()
        self.levels = levels
        self.base_filters = base_filters
        self.activation = nn.LeakyReLU(0.3, inplace=True)

        # ---------------------------------------------------------------------
        # Contracting (Down) Path
        # ---------------------------------------------------------------------
        self.down_convs = nn.ModuleList()
        self.down_inceptions = nn.ModuleList()
        self.pools = nn.ModuleList()

        # Record the actual # of output channels at each level
        self.down_channels = []

        in_ch = image_channels
        for lvl in range(1, levels + 1):
            out_ch = get_filters_count(lvl, base_filters)

            # Conv -> BN -> Activation
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(out_ch)
            self.down_convs.append(nn.Sequential(conv, bn, self.activation))

            # Inception
            inc = InceptionModule(in_channels=out_ch, out_channels=out_ch,
                                  activation=self.activation)
            self.down_inceptions.append(inc)

            # Actual output channels after Inception
            actual_out_ch = (out_ch // 4) * 4
            self.down_channels.append(actual_out_ch)

            # MaxPool except at the last (bottom) level
            if lvl < levels:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.pools.append(None)

            # Update input channels for the next level
            in_ch = actual_out_ch

        # ---------------------------------------------------------------------
        # Expanding (Up) Path
        # ---------------------------------------------------------------------
        self.up_convs = nn.ModuleList()
        self.up_inceptions = nn.ModuleList()

        for lvl in range(levels, 1, -1):
            # Skip connection channels from contracting path
            skip_channels = self.down_channels[lvl - 2]
            # Bottom (upsampled) feature channels from the contracting path
            bottom_channels = self.down_channels[lvl - 1]
            # Total input channels = skip + bottom
            total_channels = bottom_channels + skip_channels

            # Create convolutional layers for expanding path
            up_conv = nn.Conv2d(total_channels, skip_channels, kernel_size=3, padding=1)
            bn_up = nn.BatchNorm2d(skip_channels)
            self.up_convs.append(nn.Sequential(up_conv, bn_up, self.activation))

            # Create InceptionModule for expanding path
            up_inception = InceptionModule(in_channels=skip_channels, out_channels=skip_channels,
                                           activation=self.activation)
            self.up_inceptions.append(up_inception)

        # ---------------------------------------------------------------------
        # Final: 1×1 or 3×3 conv to 1 channel
        # ---------------------------------------------------------------------
        self.final_conv = nn.Conv2d(self.down_channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        1) Contracting path:
           for i in [0..levels-1]:
             x -> conv -> inception -> store skip
             if i < levels-1: pool
        2) Expanding path:
           for i in [levels-1..1]:
             upsample
             cat with skip
             conv + inception
        3) Final conv -> sigmoid
        """
        # -------------------------------------
        # Contracting Path
        # -------------------------------------
        downs = []
        curr = x
        for i in range(self.levels):
            curr = self.down_convs[i](curr)       # Conv
            curr = self.down_inceptions[i](curr) # Inception
            downs.append(curr)
            if self.pools[i] is not None:
                curr = self.pools[i](curr)

        # -------------------------------------
        # Expanding Path
        # -------------------------------------
        for i, (up_conv, up_inception) in enumerate(zip(self.up_convs, self.up_inceptions)):
            curr = F.interpolate(curr, scale_factor=2, mode='nearest')
            skip = downs[-(i + 2)]  # Skip connection from contracting path
            curr = torch.cat([curr, skip], dim=1)  # Concatenate along channels
            curr = up_conv(curr)  # Apply Conv -> BN -> Activation
            curr = up_inception(curr)  # Apply Inception

        # -------------------------------------
        # Final Layer
        # -------------------------------------
        curr = self.final_conv(curr)
        return curr

class ConvBlockFusion(nn.Module):
    """Convolutional block for fusion models."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate mechanism for feature fusion."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EncoderBlockFusion(nn.Module):
    """Encoder block for fusion models."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlockFusion(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class DecoderBlockFusion(nn.Module):
    """Decoder block for fusion models."""
    
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, F_int: int, use_attention: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        if use_attention:
            self.attention = AttentionGate(F_g=mid_channels, F_l=mid_channels, F_int=F_int)
        self.conv = ConvBlockFusion(mid_channels + mid_channels, out_channels)
        self.use_attention = use_attention

    def forward(self, x: torch.Tensor, combined_skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.use_attention:
            combined_skip = self.attention(x, combined_skip)
        
        x = torch.cat([x, combined_skip], dim=1)
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    """Channel attention mechanism."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pooled = self.avg_pool(x).view(b, c)
        max_pooled = self.max_pool(x).view(b, c)
        avg_out = self.fc2(self.relu(self.fc1(avg_pooled)))
        max_out = self.fc2(self.relu(self.fc1(max_pooled)))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block with attention mechanisms."""
    
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention_local = ChannelAttention(in_channels)
        self.spatial_attention_local = SpatialAttention()

        self.channel_attention_global = ChannelAttention(in_channels)
        self.spatial_attention_global = SpatialAttention()
        
        self.fusion_conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, global_feat: torch.Tensor, local_feat1: torch.Tensor, local_feat2: torch.Tensor) -> torch.Tensor:
        # Apply channel attention to each feature map
        global_ca = global_feat * self.channel_attention_global(global_feat)
        local_ca1 = local_feat1 * self.channel_attention_local(local_feat1)
        local_ca2 = local_feat2 * self.channel_attention_local(local_feat2)

        # Apply spatial attention to each feature map
        global_sa = global_ca * self.spatial_attention_global(global_ca)
        local_sa1 = local_ca1 * self.spatial_attention_local(local_ca1)
        local_sa2 = local_ca2 * self.spatial_attention_local(local_ca2)

        # Concatenate the feature maps
        fused_features = torch.cat((global_sa, local_sa1, local_sa2), dim=1)
        
        # Fuse them using a convolutional layer
        fused_features = self.fusion_conv(fused_features)
        fused_features = self.relu(fused_features)
        
        return fused_features


class SimpleFeatureFusionBlock(nn.Module):
    """Simple feature fusion block without attention."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, global_feat: torch.Tensor, local_feat1: torch.Tensor, local_feat2: torch.Tensor) -> torch.Tensor:
        # Concatenate the feature maps along the channel dimension
        fused_features = torch.cat((global_feat, local_feat1, local_feat2), dim=1)
        
        # Apply a convolutional layer to reduce dimensions
        fused_features = self.fusion_conv(fused_features)
        fused_features = self.relu(fused_features)
        
        return fused_features


class VENUS(nn.Module):
    """
    VENUS: Multi-Input UNet for breast segmentation with feature fusion.
    
    This model processes three input streams and fuses their features at multiple levels
    to improve segmentation performance. Originally called MultiInputUNet.
    """
    
    def __init__(self, n_channels: int = 1, n_classes: int = 1, use_simple_fusion: bool = False, 
                 use_decoder_attention: bool = True, base_channels: int = 64):
        super().__init__()

        if use_simple_fusion:
            self.fusion_skip1 = SimpleFeatureFusionBlock(base_channels, base_channels)
            self.fusion_skip2 = SimpleFeatureFusionBlock(base_channels * 2, base_channels * 2)
            self.fusion_skip3 = SimpleFeatureFusionBlock(base_channels * 4, base_channels * 4)
            self.fusion_skip4 = SimpleFeatureFusionBlock(base_channels * 8, base_channels * 8)
        else:
            self.fusion_skip1 = FeatureFusionBlock(base_channels, base_channels)
            self.fusion_skip2 = FeatureFusionBlock(base_channels * 2, base_channels * 2)
            self.fusion_skip3 = FeatureFusionBlock(base_channels * 4, base_channels * 4)
            self.fusion_skip4 = FeatureFusionBlock(base_channels * 8, base_channels * 8)

        # Encoders for each input stream
        self.encoder1 = nn.ModuleList([
            EncoderBlockFusion(n_channels, base_channels),
            EncoderBlockFusion(base_channels, base_channels * 2),
            EncoderBlockFusion(base_channels * 2, base_channels * 4),
            EncoderBlockFusion(base_channels * 4, base_channels * 8),
        ])
        self.encoder2 = nn.ModuleList([
            EncoderBlockFusion(n_channels, base_channels),
            EncoderBlockFusion(base_channels, base_channels * 2),
            EncoderBlockFusion(base_channels * 2, base_channels * 4),
            EncoderBlockFusion(base_channels * 4, base_channels * 8),
        ])
        self.encoder3 = nn.ModuleList([
            EncoderBlockFusion(n_channels, base_channels),
            EncoderBlockFusion(base_channels, base_channels * 2),
            EncoderBlockFusion(base_channels * 2, base_channels * 4),
            EncoderBlockFusion(base_channels * 4, base_channels * 8),
        ])

        if use_simple_fusion:
            self.deep_feature_fusion = SimpleFeatureFusionBlock(base_channels * 8, base_channels * 8)
        else:
            self.deep_feature_fusion = FeatureFusionBlock(base_channels * 8, base_channels * 8)
        
        out_channels = 16 if base_channels < 32 else 32

        # Decoder Blocks
        self.decoder1 = DecoderBlockFusion(base_channels * 8, base_channels * 8, base_channels * 4, base_channels * 4, use_attention=use_decoder_attention)
        self.decoder2 = DecoderBlockFusion(base_channels * 4, base_channels * 4, base_channels * 2, base_channels * 2, use_attention=use_decoder_attention)
        self.decoder3 = DecoderBlockFusion(base_channels * 2, base_channels * 2, base_channels, base_channels, use_attention=use_decoder_attention)
        self.decoder4 = DecoderBlockFusion(base_channels, base_channels, out_channels, out_channels, use_attention=use_decoder_attention)

        self.final_conv = nn.Conv2d(out_channels, n_classes, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        # Process each input through its respective encoders
        skips1, p1 = self.process_through_encoders(x1, self.encoder1)
        skips2, p2 = self.process_through_encoders(x2, self.encoder2)
        skips3, p3 = self.process_through_encoders(x3, self.encoder3)

        fused_skips1 = self.fusion_skip1(skips1[0], skips2[0], skips3[0])
        fused_skips2 = self.fusion_skip2(skips1[1], skips2[1], skips3[1])
        fused_skips3 = self.fusion_skip3(skips1[2], skips2[2], skips3[2])
        fused_skips4 = self.fusion_skip4(skips1[3], skips2[3], skips3[3])

        fused_features = self.deep_feature_fusion(p1, p2, p3)
        
        # Decode the combined features
        d1 = self.decoder1(fused_features, fused_skips4)
        d2 = self.decoder2(d1, fused_skips3)
        d3 = self.decoder3(d2, fused_skips2)
        d4 = self.decoder4(d3, fused_skips1)

        return self.final_conv(d4)

    def process_through_encoders(self, x: torch.Tensor, encoders: nn.ModuleList) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process input through encoder layers."""
        skips = []
        p = x
        for encoder in encoders:
            x, p = encoder(p)
            skips.append(x)
        return skips, p


def get_model(model_name: str, in_channels: int = 1, out_channels: int = 1, **kwargs) -> nn.Module:
    """Factory function to get model by name."""
    
    # Handle UNet with encoder (ResNet, etc.) - use segmentation_models_pytorch
    if model_name.lower() == 'unet' and kwargs.get('encoder_name'):
        return smp.Unet(
            encoder_name=kwargs.get('encoder_name', 'resnet34'),
            encoder_weights=kwargs.get('encoder_weights', None),
            in_channels=in_channels,
            classes=out_channels,
        )
    
    # Create models directly instead of using lambdas to avoid pickling issues
    if model_name == 'unet':
        return MonaiUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
    elif model_name == 'custom_unet':
        return UNet(n_channels=in_channels, n_classes=out_channels, **kwargs)
    elif model_name == 'swin_unetr':
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=2,
            use_v2=True,
            downsample="mergingv2"
        )
    elif model_name == 'unetplusplus':
        return BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(32, 32, 64, 128, 256, 32),
        )
    elif model_name == 'fcn_ffnet':
        return FcnnFnet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
    elif model_name == 'segnet':
        return SegNet(input_nbr=in_channels, label_nbr=out_channels)
    elif model_name == 'skinny':
        return SkinnyNet(image_channels=in_channels, levels=6, base_filters=19)
    elif model_name == 'venus':
        return VENUS(
            n_channels=in_channels, 
            n_classes=out_channels, 
            use_simple_fusion=kwargs.get('use_simple_fusion', False),
            use_decoder_attention=kwargs.get('use_decoder_attention', True),
            base_channels=kwargs.get('base_channels', 64)
        )

    else:
    
        aux_params = dict(
                    pooling = 'avg',  # one of 'avg', 'max'
                    dropout = 0.5,  # dropout ratio, default is None
                    activation = None,  # activation function, default is None
                    classes = out_channels,  # define number of output labels
                )
            
        return smp.create_model(
                    arch='UNet', encoder_name=model_name,
                        aux_params = aux_params, 
                    in_channels = in_channels)
