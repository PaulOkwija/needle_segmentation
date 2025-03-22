from obtain_model import get_model
import torch
from torchinfo import summary

def test():
    x = torch.randn((3,1,256,256))
    model = get_model('AttUnet')
    # model = UNET(in_channels=1, out_channels=1)
    print(summary(model,((3,1,256,256))))
    preds=model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__=="__main__":
    test()