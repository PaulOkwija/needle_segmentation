import torch
from torch import nn


class AAF(nn.Module):
  def __init__(self,P):
    super(AAF,self).__init__()
    self.P = P
    self.n = P.size()[0]
    self.F = [
        nn.ELU(),nn.Hardshrink(),nn.Hardtanh(),nn.LeakyReLU(),nn.LogSigmoid(),
        nn.ReLU(),nn.ReLU6(),nn.RReLU(),nn.SELU(),nn.CELU(),nn.Sigmoid(),
        nn.Softplus(),nn.Softshrink(),nn.Softsign(),nn.Tanh(),nn.Tanhshrink()
    ]
    

  def forward(self,x):
    sz = x.size()
    res = torch.zeros(sz).cuda()
    for i in range(self.n):
      res += self.P[i] * self.F[i](x)
    
    return res


class Model(nn.Module):
    def __init__(self,input_channels,output_channels,filter_bank):
        super(Model,self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        fb = filter_bank
        
        self.P = nn.Parameter(torch.ones(16,16).cuda()*(1/16))
        self.af = list()
        for i in range(16):
          self.af.append(AAF(self.P[i,:]))
        
        # Defining Network Structure
        self.l1c1 = nn.Conv2d(self.input_channels,fb[0],kernel_size=3,padding=1)
        self.l1c2 = nn.Conv2d(fb[0],fb[0],kernel_size=3,padding=1)
        self.l2c1 = nn.Conv2d(fb[0],fb[1],kernel_size=3,padding=1)
        self.l2c2 = nn.Conv2d(fb[1],fb[1],kernel_size=3,padding=1)
        self.l3c1 = nn.Conv2d(fb[1],fb[2],kernel_size=3,padding=1)
        self.l3c2 = nn.Conv2d(fb[2],fb[2],kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.mlc1 = nn.Conv2d(fb[2],fb[3],kernel_size=3,padding=1)
        self.mlc2 = nn.Conv2d(fb[3],fb[3],kernel_size=3,padding=1)

        self.up1 = nn.ConvTranspose2d(fb[3],fb[2],kernel_size=2,stride=2)
        self.r1c1 = nn.Conv2d(fb[3],fb[2],kernel_size=3,padding=1)
        self.r1c2 = nn.Conv2d(fb[2],fb[2],kernel_size=3,padding=1)
        self.up2 = nn.ConvTranspose2d(fb[2],fb[1],kernel_size=2,stride=2)
        self.r2c1 = nn.Conv2d(fb[2],fb[1],kernel_size=3,padding=1)
        self.r2c2 = nn.Conv2d(fb[1],fb[1],kernel_size=3,padding=1)
        self.up3 = nn.ConvTranspose2d(fb[1],fb[0],kernel_size=2,stride=2)
        self.r3c1 = nn.Conv2d(fb[1],fb[0],kernel_size=3,padding=1)
        self.r3c2 = nn.Conv2d(fb[0],fb[0],kernel_size=3,padding=1)

        self.output_layer = nn.Conv2d(fb[0],self.output_channels,kernel_size=1)
        self.threshold = nn.Threshold(0.3,0.0)


    def forward(self,X):
        l1c1 = self.af[0](self.l1c1(X))
        l1c2 = self.af[1](self.l1c2(l1c1))
        l1pool = self.pool(l1c2)

        l2c1 = self.af[2](self.l2c1(l1pool))
        l2c2 = self.af[3](self.l2c2(l2c1))
        l2pool = self.pool(l2c2)

        l3c1 = self.af[4](self.l3c1(l2pool))
        l3c2 = self.af[5](self.l3c2(l3c1))
        l3pool = self.pool(l3c2)

        mlc1 = self.af[6](self.mlc1(l3pool))
        mlc2 = self.af[7](self.mlc2(mlc1))

        up1 = self.up1(mlc2)
        up1 = self.concat(up1,l3c2)
        r1c1 = self.af[8](self.r1c1(up1))
        r1c2 = self.af[9](self.r1c2(r1c1))

        up2 = self.up2(r1c2)
        up2 = self.concat(up2,l2c2)
        r2c1 = self.af[10](self.r2c1(up2))
        r2c2 = self.af[11](self.r2c2(r2c1))

        up3 = self.up3(r2c2)
        up3 = self.concat(up3,l1c2)
        r3c1 = self.af[12](self.r3c1(up3))
        r3c2 = self.af[13](self.r3c2(r3c1))

        result = self.output_layer(r3c2)
        result = torch.sigmoid(result)
        result = self.threshold(result)

        return result


    def concat(self,down,left):
        res = torch.zeros(left.size()[0],left.size()[1]*2,down.size()[2],down.size()[3]).cuda()
        hs = (left.size()[2]//2)-(down.size()[2]//2)
        he = (left.size()[2]//2)+(down.size()[2]//2)
        ws = (left.size()[3]//2)-(down.size()[3]//2)
        we = (left.size()[3]//2)+(down.size()[3]//2)
        tmp = left[:,:,hs:he,ws:we]
        res = torch.cat([tmp,down],1).cuda()

        return res