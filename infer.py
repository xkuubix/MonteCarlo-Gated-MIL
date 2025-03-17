# %%
import yaml, os
import numpy as np
import torch
from model import GatedAttentionMIL
import logging
import utils
from net_utils import test
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_and_density(image, mean_attention, std_attention, ys, item):
    """
    This function generates a multi-panel plot with:
    - Reconstructed image
    - Attention map (mean)
    - Attention map uncertainty (std)
    - KDE plot of MC Dropout predictions

    Parameters:
    - image: The reconstructed image (torch tensor)
    - mean_attention: The mean attention map (torch tensor)
    - std_attention: The std attention map (torch tensor)
    - ys: MC Dropout prediction probabilities (torch tensor)
    - item: Contains target class and label information (dict)
    """
    fig = plt.figure(figsize=(10, 8))

    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.15])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title("Reconstructed Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mean_attention, cmap="gray")
    ax2.set_title("Mean Attention Map")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(std_attention, cmap="gray")
    ax3.set_title("Attention Map Uncertainty (STD)")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, :])
    mc_preds = ys.squeeze().cpu().numpy()  # Shape: (n,)
    sns.kdeplot(mc_preds, fill=True, color='b', alpha=0.6, ax=ax4)

    ax4.set_xlim(0, 1)
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.set_title(f"MC Dropout Prediction Density; GT: {item['target']['class'][0]} ({int(item['target']['label'][0].item())})")
    plt.tight_layout()
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
    for item in dataloaders['test']:
        images, targets = item['image'].to(device), item['target']['label']
        ys, As = model.mc_inference(input_tensor=images, n=config['n'], device=device)
        att_maps = patcher.reconstruct_attention_map(As.cpu(),
                                                     item['metadata']['tiles_indices'].squeeze(),
                                                     [1, config['data']['H'], config['data']['W']])
        image = patcher.reconstruct_image_from_patches(images.squeeze().cpu(),
                                                       item['metadata']['tiles_indices'].squeeze(),
                                                       [3, config['data']['H'], config['data']['W']])

        mean_attention = att_maps.mean(dim=0).squeeze()  # Shape: (H, W)
        std_attention = att_maps.std(dim=0).squeeze()    # Shape: (H, W)
        plot_attention_and_density(image, mean_attention, std_attention, ys, item)
        #TODO: Save the plot
        i += 1
        if i == 20:
            break

# %%