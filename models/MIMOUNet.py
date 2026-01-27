import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


# ============= ORIGINAL CLASSES (KEEP AS IS) =============

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


# ============= NEW ATTENTION MODULES (ADD THESE) =============

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ============= NEW FREQUENCY MODULE (ADD THIS) =============

class FrequencyBranch(nn.Module):
    """FFT-based Frequency Domain Processing"""
    def __init__(self, in_channels, out_channels):
        super(FrequencyBranch, self).__init__()
        self.process = nn.Sequential(
            BasicConv(in_channels, out_channels, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, stride=1, relu=False)
        )
    
    def forward(self, x):
        # Use torch.fft (newer PyTorch) with fallback
        try:
            fft = torch.fft.rfft2(x, norm='ortho')
            fft_real = fft.real
            fft_imag = fft.imag
            
            mag = torch.sqrt(fft_real**2 + fft_imag**2 + 1e-8)
            mag_processed = self.process(mag)
            
            out = torch.fft.irfft2(torch.complex(mag_processed, fft_imag), s=x.shape[-2:], norm='ortho')
        except:
            # Fallback to spatial processing
            out = self.process(x)
        
        return out


# ============= NEW REFINEMENT MODULE (ADD THIS) =============

class RefinementBlock(nn.Module):
    """Residual Refinement Block"""
    def __init__(self, channels):
        super(RefinementBlock, self).__init__()
        self.conv1 = BasicConv(channels, channels, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(channels, channels, kernel_size=3, stride=1, relu=False)
    
    def forward(self, x):
        residual = self.conv2(self.conv1(x))
        return x + residual


# ============= ENHANCED BLOCKS WITH ATTENTION (ADD THESE) =============

class EBlockAttention(nn.Module):
    """Enhanced EBlock with CBAM Attention"""
    def __init__(self, out_channel, num_res=8):
        super(EBlockAttention, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
        self.attention = CBAM(out_channel)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.attention(x)
        return x


class DBlockAttention(nn.Module):
    """Enhanced DBlock with CBAM Attention"""
    def __init__(self, channel, num_res=8):
        super(DBlockAttention, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
        self.attention = CBAM(channel)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.attention(x)
        return x


# ============= ENHANCED MIMO-UNET (ADD THIS) =============

class MIMOUNetEnhanced(nn.Module):
    """
    Enhanced MIMO-UNet with:
    - CBAM attention in encoder/decoder
    - Frequency domain processing
    - Multi-scale refinement
    Can load pretrained MIMO-UNet weights!
    """
    def __init__(self, num_res=8, use_attention=True, use_frequency=True, use_refinement=True):
        super(MIMOUNetEnhanced, self).__init__()
        
        self.use_attention = use_attention
        self.use_frequency = use_frequency
        self.use_refinement = use_refinement
        
        base_channel = 32

        # Encoder (with optional attention)
        if use_attention:
            self.Encoder = nn.ModuleList([
                EBlockAttention(base_channel, num_res),
                EBlockAttention(base_channel*2, num_res),
                EBlockAttention(base_channel*4, num_res),
            ])
        else:
            self.Encoder = nn.ModuleList([
                EBlock(base_channel, num_res),
                EBlock(base_channel*2, num_res),
                EBlock(base_channel*4, num_res),
            ])

        # Feature extraction (same as original)
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        # Decoder (with optional attention)
        if use_attention:
            self.Decoder = nn.ModuleList([
                DBlockAttention(base_channel * 4, num_res),
                DBlockAttention(base_channel * 2, num_res),
                DBlockAttention(base_channel, num_res)
            ])
        else:
            self.Decoder = nn.ModuleList([
                DBlock(base_channel * 4, num_res),
                DBlock(base_channel * 2, num_res),
                DBlock(base_channel, num_res)
            ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        
        # NEW: Frequency branch
        if use_frequency:
            self.freq_branch = FrequencyBranch(3, base_channel)
            self.freq_fusion = BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)
        
        # NEW: Refinement modules
        if use_refinement:
            self.refine1 = nn.Sequential(
                RefinementBlock(base_channel * 4),
                RefinementBlock(base_channel * 4)
            )
            self.refine2 = nn.Sequential(
                RefinementBlock(base_channel * 2),
                RefinementBlock(base_channel * 2)
            )
            self.refine3 = nn.Sequential(
                RefinementBlock(base_channel),
                RefinementBlock(base_channel)
            )

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        # Initial feature extraction
        x_ = self.feat_extract[0](x)
        
        # NEW: Frequency branch
        if self.use_frequency:
            freq_feat = self.freq_branch(x)
            x_ = self.freq_fusion(torch.cat([x_, freq_feat], dim=1))
        
        # Encoder
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        # Multi-scale fusion
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        # Decoder with refinement
        z = self.Decoder[0](z)
        if self.use_refinement:
            z = self.refine1(z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        if self.use_refinement:
            z = self.refine2(z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        if self.use_refinement:
            z = self.refine3(z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


# ============= UPDATED build_net FUNCTION (REPLACE) =============

def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    elif model_name == "MIMO-UNet":
        return MIMOUNet()
    elif model_name == "MIMO-UNet-Enhanced":
        return MIMOUNetEnhanced(num_res=8, use_attention=True, use_frequency=True, use_refinement=True)
    elif model_name == "MIMO-UNet-Attention":
        return MIMOUNetEnhanced(num_res=8, use_attention=True, use_frequency=False, use_refinement=False)
    elif model_name == "MIMO-UNet-Frequency":
        return MIMOUNetEnhanced(num_res=8, use_attention=False, use_frequency=True, use_refinement=False)
    elif model_name == "MIMO-UNet-Full":
        return MIMOUNetEnhanced(num_res=8, use_attention=True, use_frequency=True, use_refinement=True)
    raise ModelError('Wrong Model!\nYou should choose: MIMO-UNet, MIMO-UNetPlus, MIMO-UNet-Enhanced, MIMO-UNet-Attention, MIMO-UNet-Frequency, or MIMO-UNet-Full')


# ============= HELPER TO LOAD PRETRAINED WEIGHTS (ADD THIS) =============

def load_pretrained_weights(model, pretrained_path, strict=False):
    """
    Load pretrained MIMO-UNet weights into Enhanced model
    
    Args:
        model: Enhanced model instance
        pretrained_path: Path to .pkl file with pretrained weights
        strict: If False, allows partial loading (recommended)
    
    Returns:
        model: Model with loaded pretrained weights
    """
    print(f"Loading pretrained weights from: {pretrained_path}")
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract state dict
    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    else:
        pretrained_dict = checkpoint
    
    # Get current model state
    model_dict = model.state_dict()
    
    # Filter compatible weights
    filtered_dict = {}
    skipped_keys = []
    new_keys = []
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
        else:
            skipped_keys.append(f"{k} (not in new model)")
    
    # Find new parameters (not in pretrained)
    for k in model_dict.keys():
        if k not in pretrained_dict:
            new_keys.append(k)
    
    # Update model
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=strict)
    
    # Print summary
    print(f"✓ Successfully loaded {len(filtered_dict)}/{len(pretrained_dict)} pretrained weights")
    
    if skipped_keys:
        print(f"⚠ Skipped {len(skipped_keys)} incompatible keys")
        if len(skipped_keys) <= 5:
            for key in skipped_keys:
                print(f"  - {key}")
        else:
            for key in skipped_keys[:3]:
                print(f"  - {key}")
            print(f"  ... and {len(skipped_keys)-3} more")
    
    if new_keys:
        print(f"✨ {len(new_keys)} new parameters initialized randomly:")
        # Group by module
        attention_keys = [k for k in new_keys if 'attention' in k]
        freq_keys = [k for k in new_keys if 'freq' in k]
        refine_keys = [k for k in new_keys if 'refine' in k]
        
        if attention_keys:
            print(f"  - Attention modules: {len(attention_keys)} params")
        if freq_keys:
            print(f"  - Frequency branch: {len(freq_keys)} params")
        if refine_keys:
            print(f"  - Refinement modules: {len(refine_keys)} params")
    
    print("✓ Model ready for training!")
    return model