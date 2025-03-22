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
    parser.add_argument("--batch_size", "-bs", dest="batch_size", type=int, default=2, help="Batch_size")
    parser.add_argument("--model_dir", "-sd", type=str, dest="model_path", default="/mydrive/Notebooks/Saved_models",help="Model weights file path.")
    parser.add_argument("--arch", "---arch", type=str, dest="model", default="Unet", help="Load model with this name.")

    args = parser.parse_args()
    print("################### Arguments chosen ####################")
    print(args)
    model_dir = args.model_path
    model = args.model
    path = args.path
    csv_path = args.csv_path
    batch_size = args.batch_size
    

    
    # Extracting data from csv files to respective dataframes
    data = extract_df(csv_path) 
    human, porcine, chicken, bovine = separate_df(data,['human', 'porcine', 'chicken', 'bovine'],5)
    # Run pre-processing function or py file before the lines below
    # train_transform = A.Compose([ToTensorV2(),],)

    val_transform = A.Compose([ToTensorV2(),],)

    datasets = {'human': human, 'porcine': porcine, 'chicken': chicken, 'bovine': bovine}
    X_train, y_train, X_test, y_test, X_val, y_val, test_dict = get_images_masks_datasets(datasets, path,[0.6,0.3,0.1],True)

    num_workers = 2

    test_ds = NeedleDataset(X_test,y_test,val_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("##################### Sanity check on data loaders #######################")
    print(type(test_ds[1]))
    print(len(test_ds[3]))
    print("Image_shape: ", test_ds[3][0].shape)
    print("Mask_shape: ", test_ds[3][1].shape)

    print("##################### Loading models #######################")
    
    # Model development
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Model(num_classes=1,input_channels=1, drop_rate=0.25,img_size=shape[0]).type(torch.cuda.DoubleTensor).to(device)
    model = get_model(model).to(device)

    # loss_fn = IOULoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    
    # Testing
    print("##################### Testing all data types #######################")
    test_dice, test_iou = check_accuracy(test_loader, model, "Test", device=device)
    
    # print(f"Best Dice score: {test_dice}")
    # print(f"Best IoU score: {test_iou}")
    
    print("##################### Running tests per animal type #######################")
    for i in list(test_dict.keys()):
        test_ds = NeedleDataset(test_dict[i][0],test_dict[i][1],val_transform)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        print(type(test_ds[1]))
        print(len(test_ds[3]))
        print("Image_shape: ", test_ds[3][0].shape)
        print("Mask_shape: ", test_ds[3][1].shape)

        test_dice, test_iou = check_accuracy(test_loader, model, i, device=device)
        # print(f"Best {i} Dice score: {test_dice}")
        # print(f"Best {i} IoU score: {test_iou}")


if __name__ == "__main__":
    main()