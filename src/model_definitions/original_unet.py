import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

#Create a block of two convolutional neural networks
class DoubleConv(nn.Module):
    # Construct conv block
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,3,1,1,bias=False),       # Bias does similar thing to batch norm, set it to false because we use batch norm
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,3,1,1,bias=False),       # Bias does similar thing to batch norm, set it to false because we use batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    # Use the conv block    
    def forward(self, x):
        return self.conv(x)
class Model(nn.Module):
    def __init__(self,in_channels = 1, out_channels=1, features = [64,128,256,512]):
        super(Model,self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Decoder of the UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Encoder of the UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature,kernel_size=2,stride=2,)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        # Last central block
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        # Last conv before output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x) 
        x = self.bottleneck(x)   
        skip_connections = skip_connections[::-1]   #Reverses list 

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx + 1](concat_skip)
        return self.final_conv(x)
