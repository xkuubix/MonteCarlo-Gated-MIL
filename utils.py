from pandas import DataFrame
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.transforms import v2 as T
from dataset import BreastCancerDataset
import argparse

def get_args_parser():
    default = '/home/jr_buler/mcdo/config.yml'
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=default,
                        help=help)
    return parser


def random_split_df(df: DataFrame,
                    train_rest_frac, val_test_frac,
                    seed) -> tuple:
    train = df.sample(frac=train_rest_frac, random_state=seed)
    x = df.drop(train.index)
    val = x.sample(frac=val_test_frac, random_state=seed)
    test = x.drop(val.index)
    return train, val, test




def get_dataloaders(config):
    df = pd.read_pickle(config['data']['metadata_path'])
    seed = config['seed']
    root = config['data']['root_path']

    train_set, val_set, test_set = random_split_df(df,
                                                   config['data']['fraction_train_rest'],
                                                   config['data']['fraction_val_test'],
                                                   seed=seed)
    train_transforms = T.Compose([T.RandomHorizontalFlip(),
                                  T.RandomVerticalFlip(),
                                  T.GaussianBlur(3),
                                  ])
    train_dataset = BreastCancerDataset(root=root,
                                        df=train_set,
                                        view=config['data']['view'],
                                        is_multimodal=config['data']['multimodal'],
                                        transforms=train_transforms,
                                        bag_size=config['data']['bag_size_train'],
                                        patch_size=config['data']['patch_size'],
                                        overlap=config['data']['overlap_train'],
                                        empty_thresh=config['data']['empty_threshold'])
    val_dataset = BreastCancerDataset(root=root,
                                      df=val_set,
                                      view=config['data']['view'],
                                      is_multimodal=config['data']['multimodal'],
                                      transforms=T.Compose([]),
                                      bag_size=config['data']['bag_size_val_test'],
                                      patch_size=config['data']['patch_size'],
                                      overlap=config['data']['overlap_val_test'],
                                      empty_thresh=config['data']['empty_threshold'])
    test_dataset = BreastCancerDataset(root=root,
                                       df=test_set,
                                       view=config['data']['view'],
                                       is_multimodal=config['data']['multimodal'],
                                       transforms=T.Compose([]),
                                       bag_size=config['data']['bag_size_val_test'],
                                       patch_size=config['data']['patch_size'],
                                       overlap=config['data']['overlap_val_test'],
                                       empty_thresh=config['data']['empty_threshold'])
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config['training_plan']['parameters']['batch_size'],
                              shuffle=True,
                              num_workers=config['training_plan']['parameters']['num_workers'])
    val_loader = DataLoader(val_dataset,
                            batch_size=config['training_plan']['parameters']['batch_size'],
                            shuffle=False,
                            num_workers=config['training_plan']['parameters']['num_workers'])
    test_loader = DataLoader(test_dataset,
                             batch_size=config['training_plan']['parameters']['batch_size'],
                             shuffle=False,
                             num_workers=config['training_plan']['parameters']['num_workers'])
    dataloaders = {'train': train_loader,
                   'val': val_loader,
                   'test': test_loader}
    return dataloaders