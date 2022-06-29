import torch
from torch import nn
from torch.utils.data import Dataset
from . import (emnist_fs,
               omniglot_fs,
               cifar100_fs)

from torch.utils.data.distributed import DistributedSampler

#class ConcatDatasetSafe(ConcatDataset):
#    def __getattr__(self, name):        
#        return getattr(self.datasets[0], name)

class DuplicateDatasetMTimes(Dataset):
    """Cleanly duplicate a dataset M times. This is to avoid
    the massive overhead associated with data loader resetting
    for small dataset sizes, e.g. the support set which only
    has k examples per class.
    """
    def __init__(self, dataset, M):
        self.dataset = dataset
        self.N_actual = len(dataset)
        self.M = M
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx % self.N_actual)
    def __len__(self):
        return self.N_actual*self.M

def get_dataset(dataset, *args, **kwargs):
    
    str_to_class = {
        'cifar100_fs': cifar100_fs.CIFAR100FS,
        'omniglot_fs': omniglot_fs.OmniglotFS,
        'emnist_fs': emnist_fs.EMNISTFS,
    }
    
    if dataset not in str_to_class:
        raise Exception("Dataset {} not recognised".format(dataset))
    
    return str_to_class[dataset](*args, **kwargs)


def get_loader(dataset,
               batch_size,
               num_workers,
               distributed=False):
    
    if distributed:
        sampler = DistributedSampler(dataset)
        # yes, this must be set to false in order
        # for it to work.
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        drop_last=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )

    assert len(loader) > 0
    return loader, sampler
