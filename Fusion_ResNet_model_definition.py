#Define the Fusion-ResNet model

#Importing libraries
import torch
from torch import nn
import torch.optim as optim

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
    
class Fusion_ResNet(nn.Module):
    def __init__(self, in_channels, resblock):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),           
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(32, 64, downsample=True),
            resblock(64, 64, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.decoder_1 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64,  kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        
        self.decoder_2 = nn.Sequential(nn.ConvTranspose2d(in_channels=64*2, out_channels=32,  kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU())
        
         
        self.conv = nn.Sequential(
              nn.Conv2d(32*2, 13, kernel_size=3, stride=1, padding=1),           
              nn.Sigmoid()
          )    

    def forward(self, input):
        input = self.layer0(input)
        input1 = self.layer1(input)
        input2 = self.layer2(input1)        
        input3 = self.layer3(input2)           
        input4 = self.decoder_1(input3)        
        cat1=torch.cat([input4, input2],1)       
        input6 = self.decoder_2(cat1)          
        cat2=torch.cat([input6, input1],1)        
        input7=self.conv(cat2)      

        return input7

