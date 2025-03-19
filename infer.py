# %%
import yaml, os, shutil
import numpy as np
import torch
from model import MultiHeadGatedAttentionMIL
import logging
import utils
from net_utils import test
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms as T


def plot_attention_and_density(image, pos_att, neg_att, probs, item, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.3])


    neg_att_scaling_factor = probs[:,:,0].mean(dim=(0,1)).item()
    pos_att_scaling_factor = probs[:,:,1].mean(dim=(0,1)).item()



    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title("Input Image")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(neg_att * neg_att_scaling_factor, cmap="Blues", vmin=0., vmax=1.)
    ax2.set_title("Normal Attention")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pos_att * pos_att_scaling_factor, cmap="Reds", vmin=0., vmax=1.)
    ax3.set_title("Cancer Attention")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, :])

    mc_preds = probs.squeeze().cpu().numpy()

    positive_probs = mc_preds[:, 1]
    negative_probs = mc_preds[:, 0]

    sorted_indices = np.argsort(positive_probs)
    positive_probs_sorted = positive_probs[sorted_indices]
    negative_probs_sorted = negative_probs[sorted_indices]

    bars = np.arange(len(positive_probs_sorted))  # X-axis positions
    bar_width = 1.0

    ax4.bar(bars,
            negative_probs_sorted,
            bottom=positive_probs_sorted,
            color="royalblue",
            label="Negative (Normal)",
            width=bar_width,
            )
    ax4.bar(bars,
            positive_probs_sorted,
            color="orangered",
            label="Positive (Cancer)",
            width=bar_width,
            edgecolor='black',
            linewidth=0.5
            )

    ax4.axhline(
        0.5,
        color="gold",
        linestyle="-",
        lw=1.5,
        label="Threshold (p=0.5)"
        )

    x_min = bars[0] - 0.5 * bar_width
    x_max = bars[-1] + 0.5 * bar_width
    ax4.set_xlim(x_min, x_max)

    x_middle = (x_min + x_max) / 2
    ax4.axvline(
        x=x_middle,
        color="gold",
        linestyle="--",
        lw=1.5,
        label="n/2"
        )

    ax4.set_xlim(bars[0] - 0.5 * bar_width, bars[-1] + 0.5 * bar_width)
    ax4.set_xlabel(f"n={len(positive_probs_sorted)}")

    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Probability", fontsize=10)
    ax4.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
        )
    mean_pred = np.mean(positive_probs)
    median_pred = np.median(positive_probs)
    std_pred = np.std(positive_probs)
    iqr_pred = np.percentile(positive_probs, 75) - np.percentile(positive_probs, 25)
    min_pred, max_pred = np.min(positive_probs), np.max(positive_probs)

    stats_text = (
        f"Probability of Cancer:    "
        f"Mean: {mean_pred:.2f}    "
        f"Std: {std_pred:.2f}    "
        f"Median: {median_pred:.2f}    "
        f"IQR: {iqr_pred:.2f}    "
        f"Min: {min_pred:.2f}    Max: {max_pred:.2f}"
    )

    props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    ax4.text(0.5, 0.95, stats_text, fontsize=9, 
            verticalalignment='top', horizontalalignment='left', bbox=props)

    ax4.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        fontsize=10,
        frameon=False
        )
    title = (
        f"Positive and Negative Attentions for {len(positive_probs_sorted)} Monte Carlo Dropout Samples \n"
        f"Ground Truth: {item['target']['class'][0]}"
    )
    fig.suptitle(title)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=1_000)
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