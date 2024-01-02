import time
import warnings
import numpy as np
import torch
import open_earth_map as oem
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import os
import random
import json
from torch.optim import lr_scheduler

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="/work/remote_sensing/open_earth_map/OpenEarthMap_Mini")
parser.add_argument("--img_size", type=int , default=512)
parser.add_argument("--N_CLASSES", type=int, default=9)
parser.add_argument("--batch_size", '-b', type=int, default=4)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--loss", type=str, default="CE")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--scheduler", type=str, default=None)

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepare_data(data_dir, img_size, n_classes):
    TRAIN_LIST = os.path.join(data_dir, "train.txt")
    VAL_LIST = os.path.join(data_dir, "val.txt")
    
    fns = [f for f in Path(data_dir).rglob("*.tif") if "/images/" in str(f)]
    train_fns = [str(f) for f in fns if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
    val_fns = [str(f) for f in fns if f.name in np.loadtxt(VAL_LIST, dtype=str)]

    train_augm = torchvision.transforms.Compose(
    [
        oem.transforms.Rotate(),
        oem.transforms.Crop(img_size),
    ],
)

    val_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Resize(img_size),
        ],
    )
    
    train_data = oem.dataset.OpenEarthMapDataset(
    train_fns,
        n_classes=n_classes,
        augm=train_augm,
    )

    val_data = oem.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=n_classes,
        augm=val_augm,
    )
    return train_data, val_data

def prepare_loss(loss_name):
    if loss_name == "Jaccard":
        criterion = oem.losses.JaccardLoss()
    elif loss_name == "Dice":
        criterion = oem.losses.DiceLoss()
    elif loss_name == "CE":
        criterion = oem.losses.CEWithLogitsLoss()
    elif loss_name == "Focal":
        criterion = oem.losses.FocalLoss()
    elif loss_name == "MCC":
        criterion = oem.losses.MCCLoss()
    elif loss_name == "OHEMBCELoss":
        criterion = oem.losses.OHEMBCELoss()
    else:
        raise AttributeError(f"{loss_name} is not supported!")
    return criterion

def main():
    args = parser.parse_args()
    print("{}".format(args).replace(', ', ',\n'))
    seed_all(42)
    
    os.makedirs(args.out_dir, exist_ok=True)
    # Dataset preparation
    train_data, val_data = prepare_data(args.data_dir, 
                                        args.img_size,
                                        args.N_CLASSES)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=3,
        shuffle=True,
        drop_last=True,
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=3,
        shuffle=False,
    )
    
    # prepare the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model == "unet":
        model = oem.networks.UNet(in_channels=3, n_classes=args.N_CLASSES)
    elif args.model == "unetformer":
        model = oem.networks.UNetFormer(in_channels=3, n_classes=args.N_CLASSES)
    elif args.model == "segformer":
        model = oem.networks.Segformer(
            dims = (64, 128, 320, 512),      # dimensions of each stage
            heads = (1, 2, 5, 8),           # heads of each stage
            ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
            reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
            num_layers = 2,                 # num layers of each stage
            decoder_dim = 256,              # decoder dimension
            num_classes = args.N_CLASSES    # number of segmentation classes
        )
    else:
        raise AttributeError(f"{args.model} is not supported!")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = prepare_loss(args.loss)
    
    if args.scheduler is not None:
        if args.scheduler == "step_lr":
            scheduler = lr_scheduler.StepLR(optimizer, args.epoch // 3, 0.1)
        else:
            raise AttributeError(f"{args.scheduler} is not supported!")
    # train the model
    start = time.time()

    max_score = 0

    for epoch in range(args.epoch):
        print(f"\nEpoch: {epoch + 1}")

        train_logs = oem.runners.train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=device,
        )

        valid_logs = oem.runners.valid_epoch(
            model=model,
            criterion=criterion,
            dataloader=val_data_loader,
            device=device,
        )
        
        if args.scheduler is not None:
            scheduler.step()
            
        epoch_score = valid_logs["Score"]
        if max_score < epoch_score:
            max_score = epoch_score
            oem.utils.save_model(
                model=model,
                epoch=epoch,
                best_score=max_score,
                model_name="model.pth",
                output_dir=args.out_dir,
            )
        valid_logs["training_loss"] = train_logs["Loss"]
        valid_logs["training_score"] = train_logs["Score"]
        valid_logs["epoch"] = epoch+1
        valid_logs["lr"] = optimizer.param_groups[0]['lr']
        with open(os.path.join(args.out_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(valid_logs) + "\n")
    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
    
    with open(os.path.join(args.out_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f)
    

if __name__=="__main__":
    main()