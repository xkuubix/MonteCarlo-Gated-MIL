# %%
import yaml, os, shutil
import numpy as np
import torch
from model import GatedAttentionMIL
import logging
import utils
from net_utils import test
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms as T


def plot_attention_and_density(image, mean_attention, std_attention, ys, item, save_path=None):
    """
    Generates a multi-panel plot with:
    - Reconstructed image
    - Attention map (average)
    - Attention uncertainty (STD)
    - KDE plot of MC Dropout predictions with clinical enhancements

    Parameters:
    - image: The reconstructed image (torch tensor)
    - mean_attention: The mean attention map (torch tensor)
    - std_attention: The std attention map (torch tensor)
    - ys: MC Dropout prediction probabilities (torch tensor)
    - item: Contains target class and label information (dict)
    - save_path: Path to save the figure (optional)
    """
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mean_attention, cmap="gray")
    ax2.set_title("Average Attention Map")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(std_attention, cmap="gray")
    ax3.set_title("Attention Uncertainty (STD)")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, :])
    mc_preds = ys.squeeze().cpu().numpy()

    sns.kdeplot(mc_preds, fill=True, color='b', alpha=0.6, ax=ax4)

    ax4.axvline(0.5, color='r', linestyle='--', lw=2, label="Threshold (0.5)")

    ax4.set_xlim(0, 1)
    ax4.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.set_title(f"MC Dropout Prediction Density; GT: {item['target']['class'][0]} ({int(item['target']['label'][0].item())})")

    mean_pred = np.mean(mc_preds)
    median_pred = np.median(mc_preds)
    iqr_pred = np.percentile(mc_preds, 75) - np.percentile(mc_preds, 25)
    min_pred, max_pred = np.min(mc_preds), np.max(mc_preds)

    stats_text = (
        f"Mean: {mean_pred:.2f}     "
        f"Median: {median_pred:.2f}     "
        f"IQR: {iqr_pred:.2f}     "
        f"Min: {min_pred:.2f}     Max: {max_pred:.2f}"
    )
    ax4.set_ylim(0, ax4.get_ylim()[1] * 1.25)
    props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    x_center = (ax4.get_xlim()[0] + ax4.get_xlim()[1]) / 2
    ax4.text(x_center, ax4.get_ylim()[1] * 0.95, stats_text, fontsize=10, 
             verticalalignment='top', horizontalalignment='center', bbox=props)


    ax4.text(0.02, ax4.get_ylim()[1] * 1.05, "Normal", color="black", fontsize=12, fontweight='bold')
    ax4.text(0.9, ax4.get_ylim()[1] * 1.05, "Cancer", color="black", fontsize=12, fontweight='bold')

    ax4.grid(axis='x', linestyle=":", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=2_000)
    plt.show()



#TODO
# MCDO / MCBN + MCDO

def deactivate_batchnorm(net):
    if isinstance(net, nn.BatchNorm2d):
        net.track_running_stats = False
        net.running_mean = None
        net.running_var = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = utils.get_args_parser()
    args, unknown = parser.parse_known_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    selected_device = config['device']
    device = torch.device(selected_device if torch.cuda.is_available() else "cpu")
    
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    run = None
    dataloaders = utils.get_dataloaders(config)

    model = GatedAttentionMIL(backbone=config['model'],
                              feature_dropout=config['feature_dropout'],
                              attention_dropout=config['attention_dropout'])
    model.apply(deactivate_batchnorm)
    model_path = os.path.join(config['model_path'], config['model_id'])
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    # test(model, dataloaders['test'], device, run)
    patcher = dataloaders['test'].dataset.patcher
    i = 0
    
    folder_path = os.path.join(config['data']['root_path'], 'figures')
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    for item in dataloaders['test']:
        images, targets = item['image'].to(device), item['target']['label']
        ys, As = model.mc_inference(input_tensor=images, n=config['n'], device=device)
        att_maps = patcher.reconstruct_attention_map(As.cpu(),
                                                     item['metadata']['tiles_indices'].squeeze(),
                                                     [1, config['data']['H'], config['data']['W']])
        os.chdir(os.path.join(dataloaders['test'].dataset.root,
                              dataloaders['test'].dataset.class_name[item['metadata']['index']]))
        if config['data']['multimodal']:
            image, _ = dataloaders['test'].dataset.load_dcm_multimodal(item['metadata']['index'])
        else:
            image = dataloaders['test'].dataset.load_dcm_unimodal(item['metadata']['index'],
                                                                  image_only=True)
        if item['metadata']["laterality"][0] == 'R':
            t = T.RandomHorizontalFlip(p=1.0)
            image = t(image)
        # image = patcher.reconstruct_image_from_patches(images.squeeze().cpu(),
        #                                                item['metadata']['tiles_indices'].squeeze(),
        #                                                [3, config['data']['H'], config['data']['W']])

        mean_attention = att_maps.mean(dim=0).squeeze()  # Shape: (H, W)
        std_attention = att_maps.std(dim=0).squeeze()    # Shape: (H, W)
        i += 1
        save_path = os.path.join(folder_path, "".join(str(i)) + "_" + item['metadata']['patient_id'][0])
        plot_attention_and_density(image, mean_attention, std_attention, ys, item, save_path)
        print(f"done: {i}/{len(dataloaders['test'])}")
        # if i == 40:
        #     break
    print("FINISHED")

# %%