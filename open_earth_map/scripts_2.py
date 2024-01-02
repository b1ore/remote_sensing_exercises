import subprocess

def train(img_size: int,
          n_classes: int,
          batch_size:int,
          model: str,
          loss: str,
          epoch: int,
          lr: float=0.0001,
          scheduler=None):
    prefix_cmd = "python main_train.py"
    output_dir = f"/work/remote_sensing/outputs/{model}/{loss}/b{batch_size}_lr{lr}_s{scheduler}"
    cmd = " ".join([prefix_cmd, 
                    "--img_size", str(img_size),
                    "--N_CLASSES", str(n_classes),
                    "--batch_size", str(batch_size),
                    "--epoch", str(epoch),
                    "--out_dir", output_dir,
                    "--model", model,
                    "--loss", loss,
                    "--lr", str(lr)])
    
    if scheduler is not None:
        cmd = " ".join([cmd,
                        "--scheduler", scheduler])
        
    print("Running the command: \n", cmd)
    subprocess.run(cmd, shell=True)

def train_smp(img_size: int,
          n_classes: int,
          batch_size:int,
          encoder: str,
          decoder: str,
          loss: str,
          epoch: int,
          lr: float=0.0001,
          scheduler=None):
    prefix_cmd = "python main_train_smp.py"
    output_dir = f"/work/remote_sensing/outputs/{encoder}_{decoder}/{loss}/{img_size}/b{batch_size}_lr{lr}_s{scheduler}"
    cmd = " ".join([prefix_cmd, 
                    "--img_size", str(img_size),
                    "--N_CLASSES", str(n_classes),
                    "--batch_size", str(batch_size),
                    "--epoch", str(epoch),
                    "--out_dir", output_dir,
                    "--encoder", encoder,
                    "--decoder", decoder,
                    "--loss", loss,
                    "--lr", str(lr)])
    
    if scheduler is not None:
        cmd = " ".join([cmd,
                        "--scheduler", scheduler])
        
    print("Running the command: \n", cmd)
    subprocess.run(cmd, shell=True)

def main():
    # for loss in ["Jaccard", "CE"]:
    #     train(512, 9, 8, "unetformer_swin_vit", loss, 90, scheduler="step_lr")
    # for loss in ["Jaccard", "CE"]:
    #     for decoder in ["FPN", "unet"]:
    #         for encoder in ["timm-regnetx_064"]:
    #             train_smp(512, 9, 8, encoder, decoder, loss, 120, scheduler="step_lr")
    #             train_smp(512, 9, 8, encoder, decoder, loss, 120, scheduler="cosine")
    for loss in ["Jaccard"]:
        for decoder in ["FPN", "unet"]:
            for encoder in ["mit_b2"]:
                train_smp(256, 9, 8, encoder, decoder, loss, 120, scheduler="cosine")
                
    # train_smp(512, 9, 8, "mit_b2", "FPN", "Jaccard", 120, scheduler="step_lr")
    
if __name__ == "__main__":
    main()