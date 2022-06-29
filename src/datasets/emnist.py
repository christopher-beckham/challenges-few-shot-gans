import torch
from torchvision import datasets
import numpy as np
import string
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from torchvision.datasets.utils import (download_url, 
                                        download_and_extract_archive, 
                                        extract_archive,
                                        verify_str_arg)

class EMNIST(datasets.MNIST):
    """`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``EMNIST/processed/training.pt``
            and  ``EMNIST/processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # Updated URL from https://www.nist.gov/node/1298471/emnist-dataset since the
    # _official_ download link
    # https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download
    # is (currently) unavailable
    url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
    # Merged Classes assumes Same structure for both uppercase and lowercase version
    _merged_classes = {'c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z'}
    _all_classes = set(string.digits + string.ascii_letters)
    classes_split_dict = {
        'byclass': sorted(list(_all_classes)),
        'bymerge': sorted(list(_all_classes - _merged_classes)),
        'balanced': sorted(list(_all_classes - _merged_classes)),
        'letters': ['N/A'] + list(string.ascii_lowercase),
        'digits': list(string.digits),
        'mnist': list(string.digits),
    }

    def __init__(self, root: str, split: str, **kwargs: Any) -> None:
        self.split = verify_str_arg(split, "split", self.splits)
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)
        self.classes = self.classes_split_dict[self.split]

    @staticmethod
    def _training_file(split) -> str:
        return 'training_{}.pt'.format(split)

    @staticmethod
    def _test_file(split) -> str:
        return 'test_{}.pt'.format(split)

    def download(self) -> None:
        """Download the EMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        print('Downloading and extracting zip archive')
        download_and_extract_archive(self.url, download_root=self.raw_folder, filename="emnist.zip",
                                     remove_finished=True, md5=self.md5)
        gzip_folder = os.path.join(self.raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                extract_archive(os.path.join(gzip_folder, gzip_file), gzip_folder)

        # process and save as torch files
        for split in self.splits:
            print('Processing ' + split)
            training_set = (
                read_image_file(os.path.join(gzip_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(gzip_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
            )
            test_set = (
                read_image_file(os.path.join(gzip_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(gzip_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
            )
            with open(os.path.join(self.processed_folder, self._training_file(split)), 'wb') as f:
                torch.save(training_set, f)
            with open(os.path.join(self.processed_folder, self._test_file(split)), 'wb') as f:
                torch.save(test_set, f)
        shutil.rmtree(gzip_folder)

        print('Done!')

if __name__ == '__main__':
    ds = EMNIST("/tmp", download=1, split='bymerge')
    print(ds.class_to_idx)