import torch
import torchvision
from obtain_model import get_model

# Model development
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Model(num_classes=1,input_channels=1, drop_rate=0.25,img_size=shape[0]).type(torch.cuda.DoubleTensor).to(device)

model_dir = 'Unet.pth'

model = get_model('UNet').to(device)
model.load_state_dict(torch.load(model_dir))
model.eval()
with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")