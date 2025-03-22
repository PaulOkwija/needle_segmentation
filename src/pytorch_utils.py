import torch
import time
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from tqdm.notebook import tqdm
import wandb


## Dataset preparations
class NeedleDataset(torch.utils.data.Dataset):
    def __init__(self, image, mask,transforms=None):
        self.images = image
        self.masks = mask
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        if self.transforms is not None:
            mask = mask.squeeze()
            # mask = np.expand_dims(mask,0)
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


## Loss functions
class IOULoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        iou = (intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) - intersection.sum(1) + smooth)
        iou = 1 - iou.sum() / num
        return iou

class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


## Model functions
def train_fn(loader, model, optimizer, loss_fn, scaler, epoch,device):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)
        # print(targets.shape)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            # wandb.log({"Epoch" : epoch + 1, "epoch loss": loss})

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

##############################################################
def check_accuracy(loader, model, d_type, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    pre_time = []
    inf_time = []
    post_time = []
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            # print(x.shape)
            t1 = time_sync()
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            pre_time.append(time_sync()-t1)
            # print(x.shape)

            t2 = time_sync()
            preds = torch.sigmoid(model(x))
            inf_time.append(time_sync()-t2)

            t3 = time_sync()
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            iou_score += ((preds * y).sum()) / (
                (preds + y).sum() - (preds * y).sum()  + 1e-8
            )
            post_time.append(time_sync()-t3)

    print(
        f"\n Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"\n {d_type} Dice score: {dice_score/len(loader)}")
    print(f"{d_type} IoU score: {iou_score/len(loader)}")
    print(f"{d_type} Inference time per image: {sum(inf_time)/len(inf_time)}")
    
    model.train()
    return dice_score/len(loader), iou_score/len(loader)
##########################################################################
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        model.train()


def time_sync():
    '''Waits for all kernels in all streams on a CUDA device to complete if cuda is available.'''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# def save_checkpoint(state, filename="/content/previous_dataprep.pth"):
#     print("=> Saving checkpoint")
#     torch.save(model.state_dict(), filename)