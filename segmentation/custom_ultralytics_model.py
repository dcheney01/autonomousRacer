import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels,
                                 eps=0.001,
                                 momentum=0.03,
                                 affine=True,
                                 track_running_stats=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(Bottleneck, self).__init__()
        self.cv1 = Conv(in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1)
        self.cv2 = Conv(in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self,
                 conv1_dim=(128, 128),
                 conv2_dim=(128, 128),
                 bottleneck_channels=64,
                 num_bottlenecks=1):
        super(C2f, self).__init__()
        self.cv1 = Conv(conv1_dim[0],
                        conv1_dim[1],
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.cv2 = Conv(conv2_dim[0],
                        conv2_dim[1],
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.m = nn.ModuleList([
            Bottleneck(bottleneck_channels, bottleneck_channels)
            for _ in range(num_bottlenecks)
        ])

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()
        self.cv1 = Conv(in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.cv2 = Conv(in_channels * 2,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""
    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, kernel_size=3, stride=1)
        self.upsample = nn.ConvTranspose2d(
            c_, c_, 2, 2, 0,
            bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, kernel_size=3, stride=1)
        self.cv3 = Conv(c_, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1,
                                a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Segment(nn.Module):
    def __init__(self):
        super(Segment, self).__init__()

        self.cv2 = nn.ModuleList([
            nn.Sequential(Conv(64, 64, stride=1), Conv(64, 64, stride=1),
                          nn.Conv2d(64, 64, kernel_size=1, stride=1)),
            nn.Sequential(Conv(128, 64, stride=1), Conv(64, 64, stride=1),
                          nn.Conv2d(64, 64, kernel_size=1, stride=1)),
            nn.Sequential(Conv(256, 64, stride=1), Conv(64, 64, stride=1),
                          nn.Conv2d(64, 64, kernel_size=1, stride=1))
        ])

        self.cv3 = nn.ModuleList([
            nn.Sequential(Conv(64, 64, stride=1), Conv(64, 64, stride=1),
                          nn.Conv2d(64, 1, kernel_size=1, stride=1)),
            nn.Sequential(Conv(128, 64, stride=1), Conv(64, 64, stride=1),
                          nn.Conv2d(64, 1, kernel_size=1, stride=1)),
            nn.Sequential(Conv(256, 64, stride=1), Conv(64, 64, stride=1),
                          nn.Conv2d(64, 1, kernel_size=1, stride=1))
        ])

        self.dfl = DFL()

        self.proto = Proto(64, 64)

        self.cv4 = nn.ModuleList([
            nn.Sequential(Conv(64 * (2**i), 32, stride=1),
                          Conv(32, 32, stride=1),
                          nn.Conv2d(32, 32, kernel_size=1, stride=1))
            for i in range(3)
        ])

    def forward(self, x):
        outs = []
        proto_x = self.proto['cv1'](x)
        outs.append(self.proto['cv3'](self.proto['cv2'](
            self.proto['upsample'](proto_x))))

        for i, (cv2, cv3, cv4) in enumerate(zip(self.cv2, self.cv3, self.cv4)):
            x = cv2(x)
            proto_x = torch.cat([proto_x, x], dim=1)
            outs.append(cv3(x))
            if i != 2:
                proto_x = self.proto['upsample'](proto_x)
                proto_x = self.proto['cv2'](proto_x)
                proto_x = self.proto['cv3'](proto_x)
            outs.append(cv4(proto_x))
        return outs


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.model = nn.Sequential(Conv(3, 16), Conv(16, 32),
                                   C2f((32, 32), (48, 32), 16,
                                       1), Conv(32, 64),
                                   C2f((64, 64), (128, 64), 32, 2),
                                   Conv(64, 128),
                                   C2f((128, 128), (256, 128), 64, 2),
                                   Conv(128, 256),
                                   C2f((256, 256), (384, 256), 128, 1),
                                   SPPF(256, 128),
                                   nn.Upsample(scale_factor=2, mode='nearest'),
                                   Concat(), C2f((384, 128), (192, 128), 64,
                                                 1),
                                   nn.Upsample(scale_factor=2, mode='nearest'),
                                   Concat(), C2f((192, 64), (96, 64), 32, 1),
                                   Conv(64, 64), Concat(),
                                   C2f((192, 128), (192, 128), 64, 1),
                                   Conv(128, 128), Concat(),
                                   C2f((384, 256), (384, 256), 128, 1),
                                   Segment())

    def forward(self, x):

        return self.model(x)


# Create an instance of the network
net = SegmentationModel()

#print net and save to txt file
with open('mine.txt', 'w') as f:
    print(net, file=f)

# Load the state dict
state_dict = torch.load(
    '/home/curtis/classes/robotic_vision/autonomousRacer/segmentation/best.pth'
)

print(len(state_dict.keys()))
print(len(list(net.state_dict().keys())))

# Load the state dict into the model
net.load_state_dict(state_dict)

# Print the network architecture
print(net)
