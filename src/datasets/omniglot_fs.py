from torchvision.datasets import Omniglot
import numpy as np
import os
from . import utils as ut
from torchvision import transforms
import torch
from PIL import Image
from . import base_dataset as bd

class OmniglotFS(bd.BaseDataset):
    def get_transform(self, input_size):
        mean, std = ut.get_mean_std(1, {"dataset": {}})
        transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return transform

    def get_transform_augment(self, input_size, **kwargs):
        min_scale = kwargs.get("min_scale", 1.0)
        rot = kwargs.get("rot", 0.0)
        mean, std = ut.get_mean_std(1, {"dataset": {}})
        transform = transforms.Compose(
            [
                transforms.RandomRotation(rot, resample=Image.BILINEAR),
                transforms.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
                # gauss_blur,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return transform

    def get_transform_finetune(self, input_size, **kwargs):
        min_scale = kwargs.get("min_scale", 1.0)
        rot = kwargs.get("rot", 0.0)

        mean, std = ut.get_mean_std(1, {"dataset": {}})
        transform = transforms.Compose(
            [
                transforms.RandomRotation(rot, resample=Image.BILINEAR),
                transforms.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return transform

    def get_dataset(self, datadir, class_split, which_set, k_shot, seed):

        dataset_bg = Omniglot(
            datadir, background=True, transform=self.transform, download=True
        )
        dataset_fg = Omniglot(
            datadir, background=False, transform=self.transform, download=True
        )

        images_fg, characters_fg = zip(*dataset_fg._flat_character_images)
        images_bg, characters_bg = zip(*dataset_bg._flat_character_images)
        paths_bg = []
        for i in range(len(images_bg)):
            image, character = dataset_bg._flat_character_images[i]
            paths_bg.append(
                os.path.join(
                    dataset_bg.target_folder, dataset_bg._characters[character], image
                )
            )
        paths_fg = []
        for i in range(len(images_fg)):
            image, character = dataset_fg._flat_character_images[i]
            paths_fg.append(
                os.path.join(
                    dataset_fg.target_folder, dataset_fg._characters[character], image
                )
            )
        paths = paths_bg + paths_fg

        max_bg = len(np.unique(characters_bg))
        characters_fg = [l + max_bg for l in characters_fg]
        targets = torch.as_tensor(list(characters_bg) + list(characters_fg))
        # uniques = targets.unique()

        # "For omniglot we randomly select 1200 categories from total of 1623
        # categories as training seen categories, and select 212 categories
        # from remaining categories as unseen."
        ind_list = self.generate_ind_list_from_split(
            targets=targets,
            class_split=class_split,
            dataset_seed=seed,
            which_set=which_set,
            k_shot=k_shot,
            num_train=1200 + 211,
            num_valid=212,
        )

        paths_ = np.asarray(paths)[ind_list]
        # Dataset is small enough so just cache it all in memory
        images = [ Image.open(path, mode="r").convert("RGB") for path in paths_ ]
        targets = torch.as_tensor(targets)[ind_list]

        assert len(images) == len(targets)

        return images, targets

    @property
    def class_to_idx(self):
        return self._class_to_idx

    @property
    def n_classes(self):
        return 1623

    @property
    def n_channels(self):
        return 1

    def load_img(self, index):
        #return Image.open(self.images[index], mode="r").convert("RGB")
        return self.images[index]
