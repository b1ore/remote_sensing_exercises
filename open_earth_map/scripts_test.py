import subprocess
import os

def test(encoder,
         decoder, 
         output_dir,
         model_dir):
    prefix_cmd = "python main_test_smp.py"
    cmd = " ".join([
        prefix_cmd,
        "--output_dir", output_dir,
        "--model_dir", model_dir,
        "--encoder", encoder,
        "--decoder", decoder
    ])
    print("Running the command: \n", cmd)
    subprocess.run(cmd, shell=True)

    
def main():
    encoder = "mit_b2"
    decoder = "unet"
    model_dir = "/work/remote_sensing/outputs/mit_b2_unet/Jaccard/newda_2/b8_lr0.0001_scosine_e600"
    output_dir = os.path.join(model_dir, "prediction")
    test(encoder, decoder, output_dir, model_dir)
    
if __name__ == "__main__":
    main()