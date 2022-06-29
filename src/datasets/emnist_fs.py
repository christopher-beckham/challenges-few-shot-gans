from torchvision import transforms
import torch

from . import utils as ut 
from PIL import Image
from . import base_dataset as bd
from .emnist import EMNIST
import torchvision.transforms.functional as tf

class Rotation:
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x):
        return tf.rotate(x, self.angle)
    def __repr__(self):
        return "DeterministicRotation({})".format(self.angle)

class HorizontalFlip:
    def __init__(self):
        pass
    def __call__(self, x):
        return tf.hflip(x)
    def __repr__(self):
        return "DeterministicHorizontalFlip()"

class EMNISTFS(bd.BaseDataset):

    def get_transform(self, input_size):
        mean, std = ut.get_mean_std(1, {'dataset': {}})
        transform = transforms.Compose([
            Rotation(-90),
            HorizontalFlip(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform

    def get_transform_augment(self, input_size, **kwargs):
        min_scale = kwargs.get('min_scale', 1.0)
        rot = kwargs.get('rot', 0.)
        # Gaussian blur
        #gauss_noise = kwargs.get('gauss_noise', None)
        #if gauss_noise is not None:
        #    gauss_blur = transforms.GaussianBlur(
        #        kernel_size=gauss_noise,
        #        sigma=2.0
        #    )
        #else:
        #    gauss_blur = transforms.Lambda(lambda im: im)
        # 
        mean, std = ut.get_mean_std(1, {'dataset': {}})
        transform = transforms.Compose([
            Rotation(-90),
            HorizontalFlip(),
            transforms.RandomRotation(rot, resample=Image.BILINEAR),
            transforms.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
            #gauss_blur,
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform

    def get_transform_finetune(self, input_size, **kwargs):
        min_scale = kwargs.get('min_scale', 1.0)
        rot = kwargs.get('rot', 0.)

        mean, std = ut.get_mean_std(1, {'dataset': {}})
        transform = transforms.Compose([
            transforms.RandomRotation(rot, resample=Image.BILINEAR),
            transforms.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform
    
    def get_dataset(self, datadir, class_split, which_set, k_shot, seed):
        d_train = EMNIST(datadir, 'balanced', train=True,  download=True)
        d_test = EMNIST(datadir, 'balanced', train=False, download=True)
        data = torch.cat([d_train.data, d_test.data], dim=0)
        targets = torch.cat([d_train.targets, d_test.targets], dim=0)

        # First, define a hard-coded split between train+val and test.
        # This is based on the details of the F2GAN paper:
        # "For emnist we randomly select 28 categories from total of 48
        # categories as training seen categories, and select 10 categories
        # from remaining categories as unseen."
        # There is some confusing shit needed to be inferred here, because
        # there are actually 47 classes, AND there would be 9 classes
        # remaining for a total of 47 classes. I make the inference below.
        ind_list = self.generate_ind_list_from_split(
            targets=targets,
            class_split=class_split,
            dataset_seed=seed,
            which_set=which_set,
            k_shot=k_shot,
            num_train=28+9,
            num_valid=10,
        )
        images = data[ind_list]
        targets = torch.as_tensor(targets)[ind_list]
        return images, targets
    
    @property
    def n_classes(self):
        return 28+9+10
    
    @property
    def class_to_idx(self):
        return self._class_to_idx
        
    def __len__(self):
        return len(self.images)
    
    def load_img(self, index):
        img_pil = self.images[index]
        img_pil = Image.fromarray(img_pil.numpy(), mode='L').convert('RGB')
        return img_pil