import numpy as np
from torch.utils.data import (TensorDataset, Dataset)
from torchvision import transforms
import torch
from tqdm import tqdm
import math
from PIL import Image

def get_sample_indices(targets, classes, k_shot, which_set=None):
    if k_shot is not None and which_set is  None:
        raise Exception("If k_shot is defined then is_support_set must be either true or false")
    
    class_to_idx = get_class_to_idx(targets, classes)
    ind_list = []
    for c in classes:
        # Get the indices, sort them, then run a deterministic
        # shuffle with seed=class_idx.
        cls_indices = np.sort(class_to_idx[int(c)].numpy(), False)
        det_rnd_state = np.random.RandomState(int(c))
        det_rnd_state.shuffle(cls_indices)

        # from shuffled induces:
        # - 0:k is the support set
        # - 0:80% is the valid set (includes support set)
        # - 80%: is the test set
        if which_set is None:
            # this is the train set
            pass
        else:
            if which_set == 'supports':
                # only want the first k instances
                cls_indices = cls_indices[0:k_shot]
            else:
                this_len_val = int(len(cls_indices)*0.8)
                # take the first 80% of cls_indices for val
                # set, or the last 20% for test set
                if which_set == 'valid':
                    cls_indices = cls_indices[0:this_len_val]
                elif which_set == 'test':
                    cls_indices = cls_indices[this_len_val::]

        # re-sort indices (probably not necessary, but I
        # want to do it)
        cls_indices_t = torch.LongTensor(np.sort(cls_indices, False))

        ind_list += [cls_indices_t]
            
        #print("{} : {}".format(which_set, len(cls_indices)))

    ind_list = torch.cat(ind_list)
    
    return ind_list

def get_mean_std(n_channels, exp_dict):
    nc = n_channels
    if exp_dict['dataset'].get('normalize', None) is None:
        mean, std = [0.5]*nc, [0.5]*nc
    elif exp_dict['dataset'].get('normalize', None) == '0-1':
        mean, std = [0]*nc, [1]*nc

    return mean, std


def get_class_to_idx(targets, classes=None):
    if classes is None:
        classes = torch.unique(targets)
    class_to_idx = {}
    for c in classes:
        class_to_idx[int(c)] = torch.where(targets == c)[0]
    return class_to_idx

class TensorDictDataset(Dataset):
    """
    TODO: make this cleaner. This has to conform to
    what our FS datasets output, otherwise collate_fn
    will complain.
    """
    def __init__(self, X, y, p, transform):
        super().__init__()
        assert len(X) == len(y) == len(p)
        self.X = X
        self.y = y
        self.p = p
        self.transform = transform
    def __getitem__(self, index):
        this_X, this_y, this_p = self.X[index], self.y[index], self.p[index]
        out_dict = {
            'images': self.transform(this_X).unsqueeze(0),
            'labels': this_y,
            'probs': this_p.unsqueeze(0),
            'images_aug': self.transform(this_X).unsqueeze(0)
        }
        return out_dict
    def __len__(self):
        return len(self.X)

class CorruptedDataset(Dataset):
    """Create a corrupted version of a dataset, where
    X% of instances are deterministically assigned an
    incorrect label.
    """
    def __init__(self, dataset, p_wrong):
        """[summary]

        Args:
            dataset ([type]): dataset
            p_wrong ([type]): what proportion of labels
            to corrupt (in range [0,1]).
        """
        super().__init__()
        if p_wrong < 0 or p_wrong > 1:
            raise Exception("p_wrong must be in [0,1]")
        self.dataset = dataset
        rnd_state = np.random.RandomState(0)
        idcs = np.arange(0, len(self.dataset))
        # Set aside p_wrong % of indices, and add
        # them into a set.
        rnd_state.shuffle(idcs)
        n_wrong = int(len(idcs)*p_wrong)
        bad_idcs = idcs[ 0:n_wrong ].tolist()
        targets = self.dataset.targets.numpy().tolist()
        rnd_state.shuffle(targets)
        targets = targets[ 0:n_wrong ]
        # create a dict mapping 'bad' indices
        # to wrong labels
        self.bad_idcs = {k:v for k,v in zip(bad_idcs, targets)}

    @property
    def n_classes(self):
        return self.dataset.n_classes

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        # If index is in bad_idcs, then go into
        # items['labels'] and corrupt it.
        if index in self.bad_idcs:
            item['labels'].mul_(0).add_(self.bad_idcs[index])
        return item

    def __len__(self):
        return len(self.dataset)

def sample_beta(batch_size, alpha):
    coefs = np.random.beta(alpha, alpha, size=(batch_size, 1))
    return torch.from_numpy(coefs).float()

def _convert_imgs_to_pil_array(X):
    X_pil = []
    for elem in X:
        this_elem_pil = np.uint8(
            ((torch.clamp(elem, -1, 1)*0.5 + 0.5)*255.).numpy()
        ).transpose(1,2,0)
        if this_elem_pil.shape[-1] == 1:
            this_elem_pil = Image.fromarray(this_elem_pil[:,:,0])
        else:
            this_elem_pil = Image.fromarray(this_elem_pil, 'RGB')
        X_pil.append(this_elem_pil)
    return X_pil


@torch.no_grad()
def get_augmented_dataset_samples(dataset,
                                  model,
                                  transform,
                                  stdev=1.0,
                                  mixup=False,
                                  n_samples_per_class=128):

    """Get an augmented version of this dataset by leveraging
    a pre-specified autoencoder model. For each class in the
    dataset, the autoencoder will generate a pre-specified
    number of samples using mixup, aggregate them all inside
    a `TensorDataset` and return a loader.

    Args:
        dataset:  TODO
        model: The mixup autoencoder.
        transform: TODO
        stdev: controls the standard deviation of the prior
            distribution. By default it is 1.0, i.e. N(0,1).

    Returns:
        [type]: [description]
    """
    if not hasattr(model, 'generate_on_batch'):
        raise Exception("`model` must have generate_on_batch() method " + \
            "to be able to generate augmented images.")

    Xs, ys, probs = [], [], []
    
    model.eval()
    
    valid_labels = np.asarray(list(dataset.class_to_idx.keys()))
    for b, c in enumerate(tqdm(dataset.class_to_idx.keys(), desc='Augmenting w/ sampling')):
        batch = dataset.getitems(dataset.class_to_idx[c])
        batch_labels = batch['labels'].to(model.rank)
        #this_images = batch['images'].to(model.rank).unsqueeze(1)
        bs = batch['images'].size(0)
        n_iters = int(math.ceil(n_samples_per_class / bs))

        this_Xs, this_ys, this_probs = [], [], []
        for _ in range(n_iters):
            if mixup:
                alpha = torch.zeros((bs, 1)).uniform_(0, 1)
                alpha_cuda = alpha.to(model.rank)
                rnd_labels = torch.LongTensor(np.random.choice(valid_labels, bs)).\
                    to(model.rank)
                batch_fakes = model.sample_mixup(
                    batch_labels,
                    rnd_labels,
                    alpha_cuda,
                    denorm=False
                )
                batch_probs = alpha*(torch.eye(dataset.n_classes)[batch_labels]) + \
                    (1-alpha)*(torch.eye(dataset.n_classes)[rnd_labels])
            else:
                batch_fakes = model.sample(batch_labels, stdev=stdev, denorm=False)
                batch_probs = torch.eye(dataset.n_classes)[batch_labels]

            this_Xs.append(batch_fakes)
            this_ys.append(batch_labels)
            this_probs.append(batch_probs)

        Xs.append(torch.cat(this_Xs, dim=0).cpu()[0:n_samples_per_class])
        ys.append(torch.cat(this_ys, dim=0).cpu()[0:n_samples_per_class])
        probs.append(torch.cat(this_probs, dim=0).cpu()[0:n_samples_per_class])

    # concatenate all onto the batch axis    
    Xs = torch.cat(Xs)

    #NOTE: this must return images in the range [0,255], as
    #if these were images originally loaded from disk with PIL.
    #The reason for this is that TensorDictDataset is intended
    #to be initialised with the same torchvision transform as
    #that of `train_dataset`.
    X_pil = _convert_imgs_to_pil_array(Xs)
    ys = torch.cat(ys).view(-1, 1)
    probs = torch.cat(probs)

    assert len(X_pil) == len(ys) == len(probs)

    dataset = TensorDictDataset(X_pil, ys, probs, transform=transform)    
    
    return dataset

@torch.no_grad()
def get_reconstructed_dataset(dataset, 
                              model, 
                              transform,
                              debug=False):
    """
    Args:
        model ([type]): The mixup autoencoder.
    Returns:
        [type]: [description]
    """
    if not hasattr(model, 'reconstruct'):
        raise Exception("`model` must have reconstruct() method " + \
            "to be able to generate augmented images.")

    X, y, probs, recon_error = [], [], [], []
    
    for c in tqdm(dataset.class_to_idx.keys(), desc='Reconstructing'):
        batch = dataset.getitems(dataset.class_to_idx[c])

        images = batch['images']
        labels = batch['labels']

        # batch['images] of size (1, k_shot, f, h, w),
        # for each class.
        #X.append(batch['images'])
        #y.append(batch['labels'][0])        
        
        # TODO fix reconstruct()
        recon = model.reconstruct(images.unsqueeze(1),
                                  labels.flatten(),
                                  use_ema=False).cpu()

        if debug:
            recon = batch['images']
        
        error = torch.mean((recon-images)**2)
        recon_error.append(error.item())

        X.append(recon.squeeze(1))
        y.append(labels)
        probs.append(torch.eye(dataset.n_classes)[labels])

    X = torch.cat(X)


    #NOTE: this must return images in the range [0,255], as
    #if these were images originally loaded from disk with PIL.
    #The reason for this is that TensorDictDataset is intended
    #to be initialised with the same torchvision transform as
    #that of `train_dataset`.
    X_pil = _convert_imgs_to_pil_array(X)
    y = torch.cat(y).view(-1, 1)
    probs = torch.cat(probs)
        
    if len(X) != len(y):
        raise Exception("X and y must have equal lengths")
    
    # X is of shape (k*n_samples, 1, f, h, w)
    # and y is of shape (k*n_samples, 1)
    dataset = TensorDictDataset(X_pil, y, probs, transform)
    
    print("Reconstruction error: {}".format(np.mean(recon_error)))
    
    return dataset