import torch
import torchvision
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from tqdm.notebook import tqdm
# import wandb
from pytorch_utils import train_fn, check_accuracy, IOULoss, BCEDiceLoss, NeedleDataset
# from model_definitions.UNeXt import Model
# from model_definitions.axial_attention_unet import Model
from model_definitions.unet_adaptive_activation import Model

from general_utils import read_image, generate_mask, get_images_masks_datasets, extract_df, separate_df, plot_samples
from obtain_model import get_model
import wandb

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, dest="path", default='../data/raw/Images', help="images path.")
    parser.add_argument("--csv_path", "-csv", type=str, dest="csv_path", default='../data/raw/data_4_code_line', help="Labels csv file path.")
    # parser.add_argument("--mask", "-mp", type=str2bool, dest="masks", default=False, help="if there are there masks in the folders")
    parser.add_argument("--shape", "-sz", type=int, dest="shape", default=256, help="image size")
    parser.add_argument("--show_samples","-ss", type=str2bool, dest="samples", default=False, help="image output width in pixels")
    parser.add_argument("--batch_size", "-bs", dest="batch_size", type=int, default=2, help="Batch_size")
    parser.add_argument("--epochs", "-ep", dest="epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--save_dir", "-sd", type=str, dest="model_path", default="../saved_models", help="Model weights file path.")
    parser.add_argument("--name", "-n", type=str, dest="model_name", default="model", help="Save model with this name.")
    parser.add_argument("--arch", "---arch", type=str, dest="model", default="UNet", help="Load model with this name.")
    
    args = parser.parse_args()
    print("################### Arguments chosen ####################")
    print(args)
    arch = args.model
    path = args.path
    model_name = args.model_name
    # split = args.split
    csv_path = args.csv_path
    shape = (args.shape, args.shape)
    show_samples = args.samples
    batch_size = args.batch_size
    epochs = args.epochs
    model_path = args.model_path


    
    # Extracting data from csv files to respective dataframes
    data = extract_df(csv_path) 
    human, porcine, chicken, bovine = separate_df(data,['human', 'porcine', 'chicken', 'bovine'],5)
    # Run pre-processing function or py file before the lines below
    train_transform = A.Compose([ToTensorV2(),],)

    val_transform = A.Compose([ToTensorV2(),],)

    datasets = {'human': human, 'porcine': porcine, 'chicken': chicken, 'bovine': bovine}
    X_train, y_train, X_test, y_test, X_val, y_val, test_dict = get_images_masks_datasets(datasets, path,[0.6,0.3,0.1],True)

    num_workers = 2

    train_ds = NeedleDataset(X_train,y_train,train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = NeedleDataset(X_val,y_val,val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print(type(train_ds[1]))
    print(len(train_ds[3]))
    print("Image_shape: ", train_ds[3][0].shape)
    print("Mask_shape: ", train_ds[3][1].shape)

    print("##################### Displaying samples images #######################")
    if show_samples:
        plot_samples(train_ds)
    
    # Model development
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Model(num_classes=1,input_channels=1, drop_rate=0.25,img_size=shape[0]).type(torch.cuda.DoubleTensor).to(device)
    model = get_model(arch).to(device)

    loss_fn = IOULoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initializing Wandb
    
    # wandb.login()
    # wandb.init(project="Modular_Needle_work_tests")

    # Training loop
    scaler = torch.cuda.amp.GradScaler()
    count_m = 0
    best_dice = 0
    best_iou = 0
    for epoch in range(epochs):
        print('-----------------------------------------------------------')
        print(f"Epoch: {epoch}")
        loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch,device)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # log_metrics(checkpoint)

        # check validation accuracy
        
        train_dice, train_iou = check_accuracy(train_loader, model, "Train", device=device)
        val_dice,val_iou = check_accuracy(val_loader, model, "Val", device=device)
        # wandb.log({"Train_IoU: " : train_iou, "Train_Dice: ": train_dice,
                    # "Val_IoU: " : val_iou, "Val_Dice: ": val_dice})



        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou

            print("\n=> Saving checkpoint")
            torch.save(model.state_dict(), "{}/{}_{}.pth".format(model_path, model_name, count_m))
            print("Model saved in: ", "{}/{}_{}.pth".format(model_path, model_name, count_m))
            # wandb.log({"Best_Val_IoU: " : best_iou, "Best_Val_Dice: ": best_dice})
    
    count_m += 1     
    print(f"\n Best Dice score: {best_dice}", f"Best IoU score: {best_iou}")
    print(f"\n Train Dice score: {train_dice}", f"Train IoU score: {train_iou}")
        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=device
        # )



if __name__ == "__main__":
    main()