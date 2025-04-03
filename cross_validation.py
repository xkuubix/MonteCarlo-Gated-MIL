# %%
import os
import yaml
import uuid
import random
import numpy as np
import torch
import torch.nn as nn
import neptune
import utils
from model import MultiHeadGatedAttentionMIL
from net_utils import train_gacc, validate, mc_validate, test, mc_test, EarlyStopping

def deactivate_batchnorm(net):
    """Deactivate BatchNorm tracking for Monte Carlo Dropout (MCDO)."""
    if isinstance(net, nn.BatchNorm2d):
        net.track_running_stats = False
        net.running_mean = None
        net.running_var = None

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = utils.get_args_parser()
    args, unknown = parser.parse_known_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    selected_device = config['device']
    device = torch.device(selected_device if torch.cuda.is_available() else "cpu")

    SEED = config['seed']
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    
    run = None
    if config["neptune"]:
        run = neptune.init_run(project="ProjektMMG/MCDO")
        run["config"] = config
        run["sys/group_tags"].add(["cross-validation"])



    k_folds = config.get("k_folds", 5)
    fold_results = []

    for fold in range(k_folds):
        print(f"\nFold {fold + 1}/{k_folds}")
        dataloaders = utils.get_fold_dataloaders(config, fold)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        model = MultiHeadGatedAttentionMIL(
            backbone=config['model'],
            feature_dropout=config['feature_dropout'],
            attention_dropout=config['attention_dropout'],
            shared_attention=config['shared_att']
        )
        model.apply(deactivate_batchnorm)
        model.to(device)

        if config['training_plan']['criterion'].lower() == 'bce':
            criterion = torch.nn.BCELoss()
        elif config['training_plan']['criterion'].lower() == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Criterion not supported")
        
        if config['training_plan']['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['training_plan']['parameters']['lr'],
                                        weight_decay=config['training_plan']['parameters']['wd'])
        elif config['training_plan']['optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config['training_plan']['parameters']['lr'],
                                        weight_decay=config['training_plan']['parameters']['wd'])
        else:
            raise ValueError("Optimizer not supported")
        
        early_stopping = EarlyStopping(patience=config['training_plan']['parameters']['patience'], neptune_run=run)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        for epoch in range(1, config['training_plan']['parameters']['epochs'] + 1):
            train_gacc(model=model, dataloader=dataloaders['train'],
                       criterion=criterion, optimizer=optimizer, device=device, neptune_run=run, epoch=epoch,
                       accumulation_steps=config['training_plan']['parameters']['grad_acc_steps'], fold_idx=fold+1)
            if config['is_MCDO']:
                val_loss = mc_validate(model=model, dataloader=dataloaders['val'],
                                       criterion=criterion, device=device, neptune_run=run,
                                       epoch=epoch, N=config['N'], fold_idx=fold+1)
            else:
                val_loss = validate(model=model, dataloader=dataloaders['val'],
                                    criterion=criterion, device=device, neptune_run=run, epoch=epoch, fold_idx=fold+1)
            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch} for fold {fold + 1}")
                break

        model_name = os.path.join(config['model_path'], f"fold_{fold + 1}_{uuid.uuid4().hex}.pth")
        torch.save(early_stopping.get_best_model_state(), model_name)
        if run is not None:
            run[f"fold_{fold + 1}/best_model_path"].log(model_name)
        model = MultiHeadGatedAttentionMIL(
            backbone=config['model'],
            feature_dropout=config['feature_dropout'],
            attention_dropout=config['attention_dropout'],
            shared_attention=config['shared_att']
            )
        model.apply(deactivate_batchnorm)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        if config['is_MCDO']:
            mc_test(model, test_loader, device, run, fold+1, config['N'])
        else:
            test(model, test_loader, device, run, fold+1)

    if run is not None:
        run.stop()

# %%