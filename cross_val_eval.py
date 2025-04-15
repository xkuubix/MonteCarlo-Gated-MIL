# %%
import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import neptune
import utils
from model import MultiHeadGatedAttentionMIL
from net_utils import test, mc_test
import time
from datetime import timedelta


def aggregate_classification_reports(reports):
    aggregated = {}
    count = len(reports)

    for label in reports[0].keys():
        if isinstance(reports[0][label], dict):
            aggregated[label] = {}
            for metric in reports[0][label]:
                aggregated[label][metric] = np.mean([report[label][metric] for report in reports])
        else:
            aggregated[label] = np.mean([report[label] for report in reports])
    
    return aggregated

def deactivate_batchnorm(net):
    """Deactivate BatchNorm tracking for Monte Carlo Dropout (MCDO)."""
    if isinstance(net, nn.BatchNorm2d):
        net.track_running_stats = False
        net.running_mean = None
        net.running_var = None

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR)

    parser = utils.get_args_parser()
    args, unknown = parser.parse_known_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    selected_device = config['device']
    device = torch.device(selected_device if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    SEED = config['seed']
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    

    project = neptune.init_project(project="ProjektMMG/MCDO")

    runs_table_df = project.fetch_runs_table(
        id=[f"MCDO-{id}" for id in range(272, 281)],
        owner="jakub-buler",
        state="inactive",  # "active" or "inactive"
        trashed=False,
        ).to_pandas()
    runs_table_df['MC-ACC'] = [None] * len(runs_table_df)
    runs_table_df['nMC-ACC'] = [None] * len(runs_table_df)
    runs_table_df['MC-REP'] = [None] * len(runs_table_df)
    runs_table_df['nMC-REP'] = [None] * len(runs_table_df)


    k_folds = config.get("k_folds", 5)
    for i in range(len(runs_table_df)):
        runs_table_df.at[i, 'MC-ACC'] = []
        runs_table_df.at[i, 'nMC-ACC'] = []
        runs_table_df.at[i, 'MC-REP'] = []
        runs_table_df.at[i, 'nMC-REP'] = []
        for fold in range(k_folds):
            print(f"[{runs_table_df['sys/id'][i]}] Fold {fold + 1}/{k_folds}")
            dataloaders = utils.get_fold_dataloaders(config, fold)
            test_loader = dataloaders['test']
            
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
        # ------------------------------------------------------------------------------------------------------------------------------
            print("="*30)
            print(f"MCDO in test")
            print("="*30)
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            start_time = time.time()  # Start timing
            mc_acc, mc_report = mc_test(model, test_loader, device, None, fold + 1, config['N'])
            elapsed_time = time.time() - start_time
            formatted_time = str(timedelta(seconds=elapsed_time))
            print(f"Time elapsed: {formatted_time}")
        # ------------------------------------------------------------------------------------------------------------------------------
            dataloaders = utils.get_fold_dataloaders(config, fold)
            test_loader = dataloaders['test']
            
            model = MultiHeadGatedAttentionMIL(
                backbone=runs_table_df['config/model'][i],
                feature_dropout=runs_table_df['config/feature_dropout'][i],
                attention_dropout=runs_table_df['config/attention_dropout'][i],
                shared_attention=runs_table_df['config/shared_att'][i]
            )
            model.apply(deactivate_batchnorm)
            model_name = runs_table_df[f'fold_{fold+1}/best_model_path'][i]
            print(f"[{runs_table_df['sys/id'][i]}] Loaded {os.path.basename(model_name)}")
            model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
            model.to(device)
        # ------------------------------------------------------------------------------------------------------------------------------
            print("="*30)
            print(f"no MCDO in test")
            print("="*30)
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            start_time = time.time()  # Start timing
            acc, report = test(model, test_loader, device, None, fold+1)
            elapsed_time = time.time() - start_time
            formatted_time = str(timedelta(seconds=elapsed_time))
            print(f"Time elapsed: {formatted_time}")
            
            runs_table_df.at[i, 'MC-ACC'].extend([mc_acc])
            runs_table_df.at[i, 'nMC-ACC'].extend([acc])
            runs_table_df.at[i, 'MC-REP'].extend([mc_report])
            runs_table_df.at[i, 'nMC-REP'].extend([report])

    for j in range(len(runs_table_df)):
        print(f"[{runs_table_df['sys/id'][j]}]", end=' ')
        mc_mean = np.array(runs_table_df.at[j, 'MC-ACC']).mean()
        mc_std =   np.array(runs_table_df.at[j, 'MC-ACC']).std()
        nmc_mean = np.array(runs_table_df.at[j, 'nMC-ACC']).mean()
        nmc_std =  np.array(runs_table_df.at[j, 'nMC-ACC']).std()
        print(f"MC-ACC   → Mean: {mc_mean:.4f}, Std: {mc_std:.4f}")
        print(f"[{runs_table_df['sys/id'][j]}]", end=' ')
        print(f"nMC-ACC  → Mean: {nmc_mean:.4f}, Std: {nmc_std:.4f}")
    
            # mc_aggregated_report = aggregate_classification_reports(runs_table_df.at[i, 'MC-REP'])
            # aggregated_report = aggregate_classification_reports(runs_table_df.at[i, 'nMC-REP'])
    
    print(runs_table_df['MC-ACC'])
    print(runs_table_df['nMC-ACC'])
    print(runs_table_df['MC-REP'])
    print(runs_table_df['nMC-REP'])
    project.stop()
# %%