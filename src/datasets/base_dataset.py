import torch
from . import utils as ut 

from PIL import Image
import numpy as np
import re

from .. import setup_logger
logger = setup_logger.get_logger(__name__)

class BaseDataset:
    def __init__(self,
                 class_split,
                 datadir,
                 seed,
                 input_size,
                 k_shot=None,
                 which_set=None,
                 return_pairs=False,
                 transform_kwargs={}):
        """[summary]

        Args:
            class_split (str): train, dev, valid, or test
            k_shot (int): how many examples per class should be in the
                dev set.
            datadir (str): data dir
            seed (int): seed
            input_size (int): size of input image (assumed width == height)
            use_clf_transform (bool, optional): if set to true, use classifier
                transforms instead. This will usually be extra transforms in the
                form of random resized crops and rotations. Defaults to False.
            clf_transform_kwargs (dict, optional): Kwargs to pass to the method
                `get_transform_clf`. This is dependent on the specific class that
                inherits from `BaseDataset`. Defaults to {}.
            return_pairs: if True, return a tensor of (bs, 2, f, h, w), where
                the 2-dim axis is a pair of +ve images.

        Raises:
            Exception: [description]
        """
            
        if class_split not in ['train', 'valid', 'test']:
            raise Exception("class_split must be either train, valid, or test")

        if k_shot is None and which_set is not None:
            raise Exception("If k_shot is not defined, then which_set should be None as well")
        if k_shot is not None and which_set is None:
            raise Exception("If k_shot is defined then which_set should not be None")

        self.transform = self.get_transform(input_size)
        self.transform_augment = self.get_transform_augment(
            input_size, 
            **transform_kwargs
        )
        
        # Load in the dataset corresponding to the class
        # split.
        images, targets = self.get_dataset(
            datadir=datadir,
            class_split=class_split,
            which_set=which_set,
            k_shot=k_shot,
            seed=seed
        )
        self.images = images
        self.targets = targets
        
        self.k_shot = k_shot
        self.datadir = datadir
        self.which_set = which_set

        self.check_inputs = False

        # main attributes
        self._class_to_idx = ut.get_class_to_idx(self.targets)
        
        self.class_split = class_split
        self.return_pairs = return_pairs   

    def __str__(self):
        # TODO fix
        return """Dataset:\n{repr}\n{split}:k={k_shot} (which_set={which_set}), N={N}\ntransform: {transform}\ntransform_augment: {transform_augment}""".format(
            repr=repr(self),
            split=self.class_split,
            k_shot=self.k_shot,
            which_set=self.which_set,
            N=len(self),
            transform=str(self.transform),
            transform_augment=str(self.transform_augment)
        )
      
    @property
    def class_to_idx(self):
        return self._class_to_idx

    def getitems(self, ind_list):
        """This method must return a dictionary with the following
        keys: 'images', 'images_aug', and 'labels'.

        Args:
            ind_list ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError("You must implement this method")

    def get_transform(self, input_size, **kwargs):
        """The transform corresponding to `images`"""
        raise NotImplementedError("You must implement this method")

    def get_transform_augment(self, input_size, **kwargs):
        """The transform corresponding to `images_aug`"""
        raise NotImplementedError("You must implement this method")

    def get_transform_finetune(self, input_size, **kwargs):
        """This transform is for GAN-generated images in finetuning mode"""
        raise NotImplementedError("You must implement this method")

    def get_dataset(self, datadir, class_split, is_support_set, k_shot, seed):
        raise NotImplementedError("You must implement this method")
    
    @property
    def n_classes(self):
        # This should return the TOTAL number of classes in
        # the dataset, i.e. train+val+test combined.
        raise NotImplementedError()

    def __len__(self):
        return len(self.targets)
    
    def torch_img_to_pil(self, img):
        # Assumes the tensor is a torch tensor
        # in range [-1, +1].
        pil_img = ((img.numpy()*0.5 + 0.5)*255).astype(np.uint8).transpose(1, 2, 0)
        if pil_img.shape[-1] == 1:
            # if greyscale, squeeze off last dimension
            pil_img = pil_img.squeeze(-1)
        return Image.fromarray(pil_img)
    
    def load_img(self, index):
        """Load an image from its index. This must return 
        a PIL image, since torchvision transforms get
        applied afterward.

        Args:
            index (int): index
        """
        raise NotImplementedError("You must implement this method")

    def generate_ind_list_from_split(self,
                                     class_split,
                                     which_set,
                                     targets,
                                     k_shot,
                                     dataset_seed,
                                     num_train,
                                     num_valid,
                                     ignore_checks=False):
        """Generate a list of indices based on the class
        split specified. Concretely, given a number of classes
        specified for train and valid,  shuffle the entire set
        of classes as a function of `dataset_seed`:
        
        ```
        uniques = targets.unique()
        shuf_uniques = shuffle(uniques, seed)
        valid_classes = shuf_uniques[-num_valid:]
        train_classes = shuf_uniques[:-num_valid]
        ```        

        Args:
            class_split ([type]): [description]
            targets ([type]): [description]
            dataset_seed ([type]): [description]
            num_train ([type]): [description]
            num_valid ([type]): [description]
            num_test ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
                
        if class_split not in ['train', 'valid']:
            raise Exception("class_split must be valid")

        if class_split == 'train' and which_set is not None:
            raise Exception("train set does not support which_set flag, this should be None")

        if which_set not in ['supports', 'valid', 'test', None]:
            raise Exception("which_set must be either 'supports', 'valid', or 'test', or None")
                
        uniques = torch.unique(targets)
        
        NUM_TRAIN = num_train
        NUM_VALID = num_valid
        
        if not ignore_checks:
            if NUM_TRAIN+NUM_VALID != len(uniques):
                raise Exception("num_train+num_valid+num_test must be == # of classes")
        
        shuffled_classes = np.random.RandomState(seed=dataset_seed).permutation(len(uniques))
        valid_uniques = shuffled_classes[-NUM_VALID:]
        train_uniques = shuffled_classes[:-NUM_VALID]
        logger.info("train classes: {} ({})".format(train_uniques, len(train_uniques)))
        logger.info("valid classes: {} ({})".format(valid_uniques, len(valid_uniques)))

        assert len(valid_uniques) + len(train_uniques) == len(shuffled_classes)

        if class_split == 'train':
            classes = train_uniques
        elif class_split == 'valid':
            classes = valid_uniques

        # k_shot flag must be set if valid or test so that get_sample_indices
        # knows to do the split correctly
        k_shot_flag = k_shot if class_split=='valid' else None
        ind_list = ut.get_sample_indices(
            targets,
            classes,
            k_shot=k_shot_flag,
            which_set=which_set
        )
        logger.info("ind list: {}".format(ind_list))
        return ind_list

    def _check_inputs(self, batch):
        # Make sure these keys exist in the dictionary
        for key in ['images', 'labels', 'probs', 'images_aug']:
            if key not in batch:
                raise Exception("key {} was not found in batch".format(key))
        images = batch['images']
        images_aug = batch['images_aug']
        labels = batch['labels']
        probs = batch['probs']
        return_pairs_dim = 3 if self.return_pairs else 1
        if not (images.ndim == 4 and images.size(0) == 1):
            raise Exception("batch['images'] must be a 4d tensor (non-batched) with size(0)==1. " + \
                "Currently, its shape is: {}".format(images.shape))
        if not (images_aug.ndim == 4 and images_aug.size(0) == return_pairs_dim):
            raise Exception("batch['images_aug'] must be a 4d tensor (non-batched) with size(0)=={}".\
                format(return_pairs_dim))
        if not (labels.ndim == 1):
            raise Exception("batch['labels'] must be a 1d tensor (non-batched) (ndim==1). " + \
                "Currently, its shape is : {}".format(labels.shape))
        if not (probs.ndim == 2 and probs.size(1) == self.n_classes):
            raise Exception("batch['probs'] must be a 2d tensor (non-batched) with size(1) == n_classes")

    def sample_idx_from_target(self, target, n=1):
        idcs_for_label = self.class_to_idx[target].numpy()
        rnd_idcs = np.random.choice(idcs_for_label, size=n)
        return rnd_idcs

    @property
    def unique_targets(self):
        return self.targets.unique()

    def __getitem__(self, index):

        batch = self.getitems([index])
        # Add a version of labels which is one-hot
        # encoded.
        batch['probs'] = torch.eye(self.n_classes)[ batch['labels'] ]

        if self.return_pairs:
            # HACK: sample another image from the same class, and
            # concat it to batch['image']
            rnd_idc = self.sample_idx_from_target(batch['labels'].item())[0]            
            pil_img2 = self.load_img(rnd_idc)
            img2_aug = self.transform_augment(pil_img2)
            batch['images_aug'] = torch.cat(
                ( batch['images_aug'], img2_aug.unsqueeze(0) ),
                dim=0
            )

        if not self.check_inputs:
            # The first time __getitem__ is invoked, run
            # a check to make sure all tensor shapes are
            # as expected. Raise an exception if anything
            # is wrong.
            self._check_inputs(batch)
            self.check_inputs = False

        # batch['images'] = as_rgb(batch['images'])
        return batch

    def getitems(self, ind_list):
        img_list, tgt_list = [], []
        img_aug_list = []
        for index in ind_list:
            
            pil_img = self.load_img(index)
            img = self.transform(pil_img)
            target = int(self.targets[index])
            img_aug = self.transform_augment(pil_img)
            if self.return_pairs:
                img_aug_again = self.transform_augment(pil_img)

            img_list += [img]
            tgt_list += [torch.as_tensor(target)]
            if self.return_pairs:
                img_aug_list += [img_aug, img_aug_again]
            else:
                img_aug_list += [img_aug]

        images = torch.stack(img_list, dim=0)
        targets = torch.stack(tgt_list, dim=0)
        images_aug = torch.stack(img_aug_list, dim=0)    

        return {
            'images': images,
            'labels': targets,
            'images_aug': images_aug
        }