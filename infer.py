# %%
import yaml, os, shutil
import numpy as np
import torch
from model import MultiHeadGatedAttentionMIL
import logging
import utils
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms as T
import neptune
import random


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
    

    project = neptune.init_project(project="ProjektMMG/MCDO")

    runs_table_df = project.fetch_runs_table(
        id=[f"MCDO-{id}" for id in range(290, 296)],
        owner="jakub-buler",
        state="inactive",
        trashed=False,
        ).to_pandas()
    k_folds = config.get("k_folds", 5)
    for i in range(len(runs_table_df)):
        for fold in range(k_folds):
            # if fold != 0:
            #     continue
            print(f"[{runs_table_df['sys/id'][i]}]", end=' ')
            print(f"\nFold {fold + 1}/{k_folds}")
            SEED = config['seed']
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.use_deterministic_algorithms(True)
            torch.set_default_dtype(torch.float32)
            dataloaders = utils.get_fold_dataloaders(config, fold)
            
            model = MultiHeadGatedAttentionMIL(
                backbone=runs_table_df['config/model'][i],
                feature_dropout=runs_table_df['config/feature_dropout'][i],
                attention_dropout=runs_table_df['config/attention_dropout'][i],
                shared_attention=runs_table_df['config/shared_att'][i]
            )
            model.apply(deactivate_batchnorm)
            model_name = runs_table_df[f'fold_{fold+1}/best_model_path'][i]
            print(f"Loaded {os.path.basename(model_name)}")
            model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
            model.to(device)

            patcher = dataloaders['test'].dataset.patcher
            j = 0
            def flush_or_create_dir(path):
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)

            sys_id = runs_table_df['sys/id'][i]
            root_path = config['data']['root_path']

            # Main folder path (e.g. .../ID1)
            main_folder = os.path.join(root_path, sys_id)
            os.makedirs(main_folder, exist_ok=True)

            # Subfolder (e.g. .../ID1/figures_f{fold})
            fold_folder = os.path.join(main_folder, f"figures_f{fold}")
            flush_or_create_dir(fold_folder)
            folder_path = fold_folder
            test_loader = dataloaders['test']
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            print("before dataloading loop")
            for id, item in enumerate(test_loader):
                print("inside dataloading loop")
                images, targets = item['image'].to(device), item['target']['label']
                ys, As = model.mc_inference(input_tensor=images, N=config['N'], device=device)
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
                
                positive_attention_map = attention_maps[:, 1, :, :, :]  # shape: (n_passes, C, H, W)
                negative_attention_map = attention_maps[:, 0, :, :, :]  # shape: (n_passes, C, H, W)

                
                mean_positive_attention = positive_attention_map.mean(dim=0).squeeze()  # Shape: (H, W)
                std_positive_attention = positive_attention_map.std(dim=0).squeeze()  # Shape: (H, W)
                mean_negative_attention = negative_attention_map.mean(dim=0).squeeze()  # Shape: (H, W)
                std_negative_attention = negative_attention_map.std(dim=0).squeeze()  # Shape: (H, W)

                j += 1
                save_path = os.path.join(folder_path, "".join(str(j)) + "_" + item['metadata']['patient_id'][0])
                
                plot_attention_and_density(image,
                                        mean_positive_attention,
                                        std_positive_attention,
                                        mean_negative_attention,
                                        std_negative_attention,
                                        probs,
                                        item,
                                        save_path)
                print(f"done: {j}/{len(dataloaders['test'])}")
                # if i == 40:
                #     break
    print("FINISHED")
    project.stop()

# %%