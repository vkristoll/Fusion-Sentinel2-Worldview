#Define the Fusion-GAN model

#Importing libraries
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

#Define the building blocks of the architecture
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
    padding=padding)

def conv_n(in_channels, out_channels, kernel_size, stride=1, padding=0):  
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 
    stride=stride, padding=padding), nn.BatchNorm2d(out_channels, 
    momentum=0.1, eps=1e-5),)
                                                        
def tconv(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, 
    padding=padding, output_padding=output_padding)
    
def tconv_n(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):   
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
    stride=stride, padding=padding, output_padding=output_padding), 
    nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),)
    
#Define the generator architecture
#gen_in_ch=number of input bands for the generator
#dim_g = number of feature maps of the first convolutional layer for the generator
#gen_out_ch = number of output bands for the generator
class Gen(nn.Module):
    def __init__(self,gen_in_ch,dim_g,gen_out_ch):
        super(Gen,self).__init__()
      
        self.n1 = conv(gen_in_ch, dim_g, 3, 1, 1) 
        self.n2 = conv_n(dim_g, dim_g*2, 4, 2, 1)
        self.n3 = conv_n(dim_g*2, dim_g*4, 4, 2, 1)    
        self.n4 = conv(dim_g*4, dim_g*4, 3, 1, 1)        
      
        self.m1 = conv_n(dim_g*4, dim_g*4, 3, 1, 1)      
        self.m2 = tconv_n(dim_g*4*2, dim_g*2, 4, 2, 1)      
        self.m3 = tconv_n(dim_g*2*2, dim_g*1, 4, 2, 1)      
        self.m4 = conv(dim_g*1*2, gen_out_ch, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        n1 = self.n1(x)
        n2 = self.n2(F.leaky_relu(n1, 0.2))
        n3 = self.n3(F.leaky_relu(n2, 0.2))      
        n4 = self.n4(F.leaky_relu(n3, 0.2))        
      
        m1 = torch.cat([self.m1(F.relu(n4)), n3], 1)      
        m2 = torch.cat([self.m2(F.relu(m1)), n2], 1)      
        m3 = torch.cat([self.m3(F.relu(m2)), n1], 1)   
        m4 = self.m4(F.relu(m3))
        
        return self.sigmoid(m4)
    
# Define the discriminator architecture
# dim_in_ch=number of input bands for the discriminator
# dim_d = number of feature maps of the first convolutional layer for the discriminator
class Disc(nn.Module):
    def __init__(self,dim_in_ch,dim_d): 
        super(Disc,self).__init__()
        self.c1 = conv(dim_in_ch, dim_d, 4, 2, 1) 
        self.c2 = conv_n(dim_d, dim_d*2, 4, 2, 1)
        self.c3 = conv_n(dim_d*2, dim_d*4, 4, 2, 1)        
        self.c4 = conv(dim_d*4, 1, 4, 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):        
        x=F.leaky_relu(self.c1(x), 0.2)
        x=F.leaky_relu(self.c2(x), 0.2)
        x=F.leaky_relu(self.c3(x), 0.2)
      
        x=self.c4(x)
        
        return self.sigmoid(x)

# from torchsummary import summary

# model=Gen().to(device)
# summary(model,input_size=(gen_in_ch,patch_size,patch_size))

# model=Disc().to(device)
# summary(model,input_size=(dim_in_ch,patch_size,patch_size)) 

