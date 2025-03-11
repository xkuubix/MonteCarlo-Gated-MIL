# %%
import yaml, uuid, os
import numpy as np
import neptune
import torch
from model import GatedAttentionMIL
import logging
import utils
from net_utils import train, validate, test, EarlyStopping

#TODO
# MCDO / MCBN + MCDO

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = utils.get_args_parser()
    args, unknown = parser.parse_known_args()
    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    selected_device = config['device']
    device = torch.device(selected_device if torch.cuda.is_available() else "cpu")
    
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if config["neptune"]:
        run = neptune.init_run(project="ProjektMMG/MCDO")
        run["config"] = config
    else:
        run = None
    model = GatedAttentionMIL(resnet_type="resnet18")
    dataloaders = utils.get_dataloaders(config)
    if config['training_plan']['criterion'].lower() == 'bce':
        criterion = torch.nn.BCELoss()
    elif config['training_plan']['criterion'].lower() == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion not supported")
    
    if config['training_plan']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training_plan']['parameters']['lr'])
    elif config['training_plan']['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['training_plan']['parameters']['lr'])
    else:
        raise ValueError("Optimizer not supported")
  

    early_stopping = EarlyStopping(patience=config['training_plan']['parameters']['patience'], nepune_run=run)


    for epoch in range(1, config['training_plan']['parameters']['epochs'] + 1):
        model = train(model=model, dataloader=dataloaders['train'],
                      criterion=criterion, optimizer=optimizer, device=device, neptune_run=run, epoch=epoch)
        val_loss, val_acc = validate(model=model, dataloader=dataloaders['val'],
                                     criterion=criterion, device=device, neptune_run=run, epoch=epoch)
        if early_stopping(val_loss, val_acc, model):
            print(f"Early stopping at epoch {epoch}")
            break
    model_name = uuid.uuid4().hex
    model_name = os.path.join(config['model_path'], model_name)
    torch.save(early_stopping.get_best_model_state(), model_name)
    if run is not None:
        run["best_model_path"].log(model_name)
    test(model, dataloaders['test'], device, run)
# %%