from pandas import DataFrame
from torch.utils.data import DataLoader
import pandas as pd
# from torchvision.transforms import v2 as T
from torchvision import transforms as T
from dataset import BreastCancerDataset
import argparse
from collections import Counter
import torch, random
import numpy as np
from sklearn.model_selection import KFold, train_test_split


def get_args_parser():
    default = '/users/project1/pt01190/mmg/MonteCarlo-Gated-MIL/config.yml'
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
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
                                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ])
    val_test_transforms = T.Compose([
                                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ])
    # val_test_transforms = None
    train_dataset = BreastCancerDataset(root=root,
                                        df=train_set,
                                        view=config['data']['view'],
                                        is_multimodal=config['data']['multimodal'],
                                        transforms=train_transforms,
                                        bag_size=config['data']['bag_size_train'],
                                        img_size=[config['data']['H'], config['data']['W']],
                                        patch_size=config['data']['patch_size'],
                                        overlap=config['data']['overlap_train'],
                                        empty_thresh=config['data']['empty_threshold'])
    val_dataset = BreastCancerDataset(root=root,
                                      df=val_set,
                                      view=config['data']['view'],
                                      is_multimodal=config['data']['multimodal'],
                                      transforms=val_test_transforms,
                                      bag_size=config['data']['bag_size_val_test'],
                                      img_size=[config['data']['H'], config['data']['W']],
                                      patch_size=config['data']['patch_size'],
                                      overlap=config['data']['overlap_val_test'],
                                      empty_thresh=config['data']['empty_threshold'])
    test_dataset = BreastCancerDataset(root=root,
                                       df=test_set,
                                       view=config['data']['view'],
                                       is_multimodal=config['data']['multimodal'],
                                       transforms=val_test_transforms,
                                       bag_size=config['data']['bag_size_val_test'],
                                       img_size=[config['data']['H'], config['data']['W']],
                                       patch_size=config['data']['patch_size'],
                                       overlap=config['data']['overlap_val_test'],
                                       empty_thresh=config['data']['empty_threshold'])
    print("Class counts per set:")
    print(f"  Train set: {Counter(train_dataset.class_name)}")
    print(f"  Validation set: {Counter(val_dataset.class_name)}")
    print(f"  Test set: {Counter(test_dataset.class_name)}")
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config['seed'])

    train_loader = DataLoader(train_dataset,
                              batch_size=config['training_plan']['parameters']['batch_size'],
                              shuffle=True,
                              num_workers=config['training_plan']['parameters']['num_workers'],
                              worker_init_fn=seed_worker,
                              generator=g
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=config['training_plan']['parameters']['batch_size'],
                            shuffle=False,
                            num_workers=config['training_plan']['parameters']['num_workers'],
                            worker_init_fn=seed_worker,
                            generator=g
                            )
    test_loader = DataLoader(test_dataset,
                             batch_size=config['training_plan']['parameters']['batch_size'],
                             shuffle=False,
                             num_workers=config['training_plan']['parameters']['num_workers'],
                            worker_init_fn=seed_worker,
                            generator=g
                            )
    dataloaders = {'train': train_loader,
                   'val': val_loader,
                   'test': test_loader}
    return dataloaders





def get_fold_dataloaders(config, fold_idx):
    """
    Splits the dataset into training, validation, and test sets for cross-validation.

    Args:
        config (dict): Configuration dictionary containing data paths and settings.
        fold_idx (int): The fold index to be used for validation.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    df = pd.read_pickle(config['data']['metadata_path'])
    seed = config['seed']
    root = config['data']['root_path']
    k_folds = config['data']['cv_folds']  # Number of cross-validation folds
    train_val_df, test_df = train_test_split(df, test_size=config['data']['fraction_test'],
                                             random_state=seed, stratify=df['class'])

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    indices = list(range(len(train_val_df)))

    train_indices, val_indices = None, None
    for i, (train_idx, val_idx) in enumerate(kf.split(indices)):
        if i == fold_idx:
            train_indices, val_indices = train_idx, val_idx
            break

    if train_indices is None or val_indices is None:
        raise ValueError(f"Invalid fold index {fold_idx}. Must be in range 0-{k_folds-1}.")

    train_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transforms = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"Patients No. TRAIN: {len(train_indices)}, VAL: {len(val_indices)}, TEST: {len(test_df)}")
    train_dataset = BreastCancerDataset(root=root,
                                        df=train_val_df.iloc[train_indices],
                                        view=config['data']['view'],
                                        is_multimodal=config['data']['multimodal'],
                                        transforms=train_transforms,
                                        bag_size=config['data']['bag_size_train'],
                                        img_size=[config['data']['H'], config['data']['W']],
                                        patch_size=config['data']['patch_size'],
                                        overlap=config['data']['overlap_train'],
                                        empty_thresh=config['data']['empty_threshold'])

    val_dataset = BreastCancerDataset(root=root,
                                      df=train_val_df.iloc[val_indices],
                                      view=config['data']['view'],
                                      is_multimodal=config['data']['multimodal'],
                                      transforms=val_test_transforms,
                                      bag_size=config['data']['bag_size_val_test'],
                                      img_size=[config['data']['H'], config['data']['W']],
                                      patch_size=config['data']['patch_size'],
                                      overlap=config['data']['overlap_val_test'],
                                      empty_thresh=config['data']['empty_threshold'])

    test_dataset = BreastCancerDataset(root=root,
                                       df=test_df,
                                       view=config['data']['view'],
                                       is_multimodal=config['data']['multimodal'],
                                       transforms=val_test_transforms,
                                       bag_size=config['data']['bag_size_val_test'],
                                       img_size=[config['data']['H'], config['data']['W']],
                                       patch_size=config['data']['patch_size'],
                                       overlap=config['data']['overlap_val_test'],
                                       empty_thresh=config['data']['empty_threshold'])

    print(f"Fold {fold_idx + 1}:")
    print_class_counts(train_dataset, val_dataset, test_dataset)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset,
                              batch_size=config['training_plan']['parameters']['batch_size'],
                              shuffle=True,
                              num_workers=config['training_plan']['parameters']['num_workers'],
                              worker_init_fn=seed_worker,
                              generator=g)

    val_loader = DataLoader(val_dataset,
                            batch_size=config['training_plan']['parameters']['batch_size'],
                            shuffle=False,
                            num_workers=config['training_plan']['parameters']['num_workers'],
                            worker_init_fn=seed_worker,
                            generator=g)

    test_loader = DataLoader(test_dataset,
                             batch_size=config['training_plan']['parameters']['batch_size'],
                             shuffle=False,
                             num_workers=config['training_plan']['parameters']['num_workers'],
                             worker_init_fn=seed_worker,
                             generator=g)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


def print_class_counts(train_dataset, val_dataset, test_dataset):
    train_class_counts = Counter(train_dataset.class_name)
    val_class_counts = Counter(val_dataset.class_name)
    test_class_counts = Counter(test_dataset.class_name)

    sorted_train_class_counts = dict(sorted(train_class_counts.items()))
    sorted_val_class_counts = dict(sorted(val_class_counts.items()))
    sorted_test_class_counts = dict(sorted(test_class_counts.items()))

    print(f"  Train set class counts: {sorted_train_class_counts}", end='')
    print(f"  (Total: {sum(sorted_train_class_counts.values())})")
    print(f"  Validation set class counts: {sorted_val_class_counts}", end='')
    print(f"  (Total: {sum(sorted_val_class_counts.values())})")
    print(f"  Test set class counts: {sorted_test_class_counts} (Fixed)", end='')
    print(f"  (Total: {sum(sorted_test_class_counts.values())})")