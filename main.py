# %%
import yaml, uuid, os
import numpy as np
import neptune
import torch
from model import GatedAttentionMIL
import logging
import utils
from net_utils import train, train_gacc, validate, test, EarlyStopping
import torch.nn as nn

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

    if config["neptune"]:
        run = neptune.init_run(project="ProjektMMG/MCDO")
        run["sys/group_tags"].add(["no-BN"])
        run["config"] = config
    else:
        run = None
    model = GatedAttentionMIL(backbone=config['model'],
                              feature_dropout=config['feature_dropout'],
                              attention_dropout=config['attention_dropout'])
    model.apply(deactivate_batchnorm)
    model.to(device)
    dataloaders = utils.get_dataloaders(config)
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

    for epoch in range(1, config['training_plan']['parameters']['epochs'] + 1):
        # train(model=model, dataloader=dataloaders['train'],
        #       criterion=criterion, optimizer=optimizer, device=device, neptune_run=run, epoch=epoch)
        train_gacc(model=model, dataloader=dataloaders['train'],
                   criterion=criterion, optimizer=optimizer, device=device, neptune_run=run, epoch=epoch,
                   accumulation_steps=config['training_plan']['parameters']['grad_acc_steps'])
        val_loss = validate(model=model, dataloader=dataloaders['val'],
                            criterion=criterion, device=device, neptune_run=run, epoch=epoch)
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break
    model_name = uuid.uuid4().hex
    model_name = os.path.join(config['model_path'], model_name)
    torch.save(early_stopping.get_best_model_state(), model_name)
    if run is not None:
        run["best_model_path"].log(model_name)
    model = GatedAttentionMIL(backbone=config['model'],
                              feature_dropout=config['feature_dropout'],
                              attention_dropout=config['attention_dropout'])
    model.apply(deactivate_batchnorm)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    test(model, dataloaders['test'], device, run)
    if run is not None:
        run.stop()
# %%