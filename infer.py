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


def plot_attention_and_density(image, pos_att, pos_std, neg_att, neg_std, probs, item, save_path=None):
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 5)


    neg_att_scaling_factor = probs[:,:,0].mean(dim=(0,1)).item()
    pos_att_scaling_factor = probs[:,:,1].mean(dim=(0,1)).item()



    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title("Input Image")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(neg_att * neg_att_scaling_factor, cmap="Blues", vmin=0., vmax=1.)
    ax2.set_title("Negative Attention")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pos_att * pos_att_scaling_factor, cmap="Reds", vmin=0., vmax=1.)
    ax3.set_title("Positive Attention")
    ax3.axis("off")

    ax5 = fig.add_subplot(gs[0, 3])
    ax5.imshow(neg_std**2, cmap="gray")
    ax5.set_title("Negative Variance")
    ax5.axis("off")
    
    ax6 = fig.add_subplot(gs[0, 4])
    ax6.imshow(pos_std**2, cmap="gray")
    ax6.set_title("Positive Variance")
    ax6.axis("off")

    mc_preds = probs.squeeze().cpu().numpy()
    positive_probs = mc_preds[:, 1]

    mean_pred = np.mean(positive_probs)
    median_pred = np.median(positive_probs)
    std_pred = np.std(positive_probs)
    iqr_pred = np.percentile(positive_probs, 75) - np.percentile(positive_probs, 25)
    min_pred, max_pred = np.min(positive_probs), np.max(positive_probs)
    
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    mean_entropy = entropy.mean()
    def interpret_entropy(H):
        if H < 0.2:
            return "very low"
        elif H < 0.4:
            return "low"
        elif H < 0.6:
            return "moderate"
        else:
            return "high"
    stats_text = (
        f"Probability of Cancer:     {mean_pred:.2f} "
        f"({std_pred:.2f}) mean (std);     "
        f"{median_pred:.2f} "
        f"({iqr_pred:.2f}) median (iqr);     " 
        f"{min_pred:.2f}-{max_pred:.2f} range;\n"
        f"Mean Entropy: {mean_entropy:.2f} ({interpret_entropy(mean_entropy)} uncertainty)"
    )

    props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    
    fig.text(0.5, -0.02, stats_text, fontsize=11, 
            verticalalignment='center', horizontalalignment='center', bbox=props)

    title = (
        f"Positive and Negative Attentions for {len(positive_probs)} Monte Carlo Dropout Samples - "
        f"Ground Truth: {item['target']['class'][0]}\n"
    )
    fig.suptitle(title)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+'.pdf', format="pdf", bbox_inches="tight", dpi=5_00)
        plt.savefig(save_path+'.png', format="png", bbox_inches="tight", dpi=5_00)
    # plt.show()
    plt.close()


# DEEEEEV
# num_mc_passes = 100  # Replace with your actual number of MC passes
# ys = torch.randn(num_mc_passes, 2)
# probs = torch.nn.functional.softmax(ys, dim=-1).unsqueeze(0)
# im = torch.zeros(3,7000,3000)
# att = torch.zeros(7000,3000)
# # plot_attention_and_density(image, pos_att, pos_std, neg_att, neg_std, probs, item, save_path=None)
# plot_attention_and_density(im,att,att,att,att,probs,None)

# %%
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
    model = MultiHeadGatedAttentionMIL(
        backbone=config['model'],
        feature_dropout=config['feature_dropout'],
        attention_dropout=config['attention_dropout'],
        shared_attention=config['shared_att']
        )
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
        probs = torch.nn.functional.softmax(ys, dim=-1)

        attention_maps = patcher.reconstruct_attention_map(
            As.cpu(),
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

        positive_attention_map = attention_maps[:, 1, :, :, :]  # shape: (n_passes, C, H, W)
        negative_attention_map = attention_maps[:, 0, :, :, :]  # shape: (n_passes, C, H, W)

        
        mean_positive_attention = positive_attention_map.mean(dim=0).squeeze()  # Shape: (H, W)
        mean_negative_attention = negative_attention_map.mean(dim=0).squeeze()  # Shape: (H, W)

        i += 1
        save_path = os.path.join(folder_path, "".join(str(i)) + "_" + item['metadata']['patient_id'][0])
        
        plot_attention_and_density(image,
                                   mean_positive_attention,
                                   mean_negative_attention,
                                   probs,
                                   item,
                                   save_path)
        print(f"done: {i}/{len(dataloaders['test'])}")
        # if i == 40:
        #     break
    print("FINISHED")

# %%