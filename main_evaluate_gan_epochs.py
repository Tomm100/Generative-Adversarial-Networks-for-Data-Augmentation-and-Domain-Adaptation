import torch_fidelity
import wandb
import os
import torch
import shutil

from config import (
    DATASET_DIR, GAN_CHECKPOINTS_DIR, GAN_EPOCHS, 
    GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D, SEED, RESULTS_DIR
)
from dataset.loader import setup_dataset
from models.wgan import Generator
from eval import generate_synthetic_images
from utils.seed import set_seed


def compute_FID_and_KID(real_dir, synth_dir, epoch, device):
    """
    Calcola FID e KID confrontando due directory e logga i risultati su WandB.
    """
    print(f"  Calcolo metriche FID e KID in corso...")
    
    metrics = torch_fidelity.calculate_metrics(
        input1=synth_dir, 
        input2=real_dir, 
        cuda=(device.type == 'cuda'), 
        isc=False, 
        fid=True, 
        kid=True, 
        verbose=False
    )

    fid_score = metrics['frechet_inception_distance']
    kid_score = metrics['kernel_inception_distance_mean']

    print(f"  📊 FID: {fid_score:.4f}")
    print(f"  📊 KID: {kid_score:.4f}")
    if wandb.run is not None:
        wandb.log({
            "Fidelity/Epoch": epoch,
            "Fidelity/FID": fid_score,
            "Fidelity/KID": kid_score
        })
        
    return fid_score, kid_score


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints_dir = '/content/drive/MyDrive/ProgettoMLVM/results_SNGAN/sngan_checkpoints/'

    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="fidelity_evaluation_over_time",
        config={"seed": SEED, "epochs": GAN_EPOCHS}
    )

    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res: return
    train_dir, _, _ = res

    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    num_gen_normal = n_train_n
    
    real_normal_dir = os.path.join(train_dir, 'NORMAL')
    eval_synth_base_dir = os.path.join(RESULTS_DIR, "fidelity_eval_synth")
    os.makedirs(eval_synth_base_dir, exist_ok=True)

    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)

    for epoch in range(10, GAN_EPOCHS + 1, 10):
        ckpt_path = os.path.join(checkpoints_dir, f'G_epoch_{epoch}.pth')
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint non trovato per l'epoca {epoch}.")
            return

        print(f"\n{'='*50}\nValutazione Epoca {epoch}\n{'='*50}")

        G.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        epoch_synth_dir = os.path.join(eval_synth_base_dir, f'epoch_{epoch}')
        if os.path.exists(epoch_synth_dir): shutil.rmtree(epoch_synth_dir)
            
        generate_synthetic_images(
            G, num_gen_normal=num_gen_normal, num_gen_pneumonia=0,
            nz=GAN_NZ, n_class=GAN_N_CLASS, device=device, syn_dir=epoch_synth_dir
        )
        synth_normal_dir = os.path.join(epoch_synth_dir, 'NORMAL')

        compute_FID_and_KID(
            real_dir=real_normal_dir,
            synth_dir=synth_normal_dir,
            epoch=epoch,
            device=device
        )

        shutil.rmtree(epoch_synth_dir)

    print("\n✅ Valutazione completa!")
    wandb.finish()

if __name__ == '__main__':
    main()
