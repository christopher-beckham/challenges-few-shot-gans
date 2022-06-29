import numpy as np
import numpy as np
import os
import torch
from torch import nn
import json
from ..fid import fid_score

from torch.utils.data import IterableDataset

from ..setup_logger import get_logger

logger = get_logger(__name__)


class FidWrapper(nn.Module):
    """Just a wrapper that conforms to the same
    interface as the Inception model used to
    compute FID.
    """

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return [self.f(x).unsqueeze(-1).unsqueeze(-1)]


def load_json_from_file(filename):
    return json.loads(open(filename, "r").read())


def extract_from_data_loader(loader, N, extract_fn=None):
    """Extract N images from the data loader, and 
    concatenate them into one large minibatch.

    Args:
        loader (torch.utils.data.DataLoader): data loader
        N (int): maximum number of examples to buffer
        extract_fn (optional): function to extract
        images from the batch returned by the loader. For
        instance, if the data loader returned a tuple whose
        first element is the image and second element is
        the label, then extract_fn should be lambda x: x[0].
    """
    buf = []
    if extract_fn is None:
        extract_fn = lambda x: x
    n_imgs_processed = 0
    for elem in loader:
        this_imgs = extract_fn(elem)
        buf.append(this_imgs)
        n_imgs_processed += this_imgs.size(0)
        if N is not None:
            if n_imgs_processed >= N:
                break
    return torch.cat(buf, dim=0)

def precompute_fid_stats(loader, batch_size, N=5000, model=None):
    imgs0 = extract_from_data_loader(
        loader, N, extract_fn=lambda dd: dd["images"][:, 0]
    )[0:N]
    fid_mean, fid_sd = fid_score.compute_statistics_given_imgs(
        imgs0, batch_size, 0, model=model
    )
    if len(imgs0) < N:
        logger.warning("len(imgs)={} < fid_N={}".format(len(imgs0), N))
    return fid_mean, fid_sd


def precompute_fid_activations(loader, batch_size, N=5000, model=None, extract_fn=None):
    if extract_fn is None:
        extract_fn = lambda dd: dd["images"][:, 0]
    imgs0 = extract_from_data_loader(
        loader, N, extract_fn=extract_fn
    )[0:N]
    acts = fid_score.get_activations(imgs0, model, batch_size, device=0)
    return acts


# def _precompute_fid_stats_per_class(dataset, batch_size, N=5000, model=None):
#    class2fid = {}
#    for label in dataset.unique_targets:
#        batch = dataset.getitems(dataset.sample_idx_from_target(label.item(), n=N))
#        images = batch["images"]
#        this_fid_mean, this_fid_sd = fid_score.compute_statistics_given_imgs(
#            images, batch_size, 0, model=model
#        )
#        class2fid[label.item()] = (this_fid_mean, this_fid_sd)
#    return class2fid


def str2bool(st):
    if st.lower() == "true":
        return True
    return False


class Argument:
    def __init__(self, name, default, types, choices=None):
        """_summary_

        Args:
            name (_type_): name of the argument
            default (_type_): its default value
            types (_type_): a list of allowed types
            choices: the only allowable values that can be taken
        """
        self.name = name
        self.default = default
        self.types = types
        self.choices = choices

    def validate(self, x):
        if type(x) not in self.types:
            raise Exception(
                "argument {} has invalid type: {}, allowed types = {}".format(
                    self.name, type(x), self.types
                )
            )
        if self.choices is not None:
            if x not in self.choices:
                raise Exception(
                    "argument {} has value {} but allowed values = {}".format(
                        self.name, x, self.choices
                    )
                )


def validate_and_insert_defaults(exp_dict, defaults, ignore_recursive=[]):
    """Inserts default values into the exp_dict.

    Will raise an exception if exp_dict contains
    a key that is not recognised. If the v for a
    (k,v) pair is also a dict, this method will
    recursively call insert_defaults() into that
    dictionary as well.

    Args:
        exp_dict (dict): dictionary to be added to
        ignore_recursive: any key in here will not be validated
          recursively.
    """
    ignore_recursive = set(ignore_recursive)

    for key in exp_dict.keys():
        if key not in defaults:
            # Check if there are any unknown keys.
            print(exp_dict)
            raise Exception(
                "Found key in exp_dict but is not recognised: {}".format(key)
            )
        else:
            if type(defaults[key]) == dict and key not in ignore_recursive:
                # If this key maps to a dict, then apply
                # this function recursively
                validate_and_insert_defaults(exp_dict[key], defaults[key])
            else:
                pass
                """
                if expand_env_vars and type(exp_dict[key]) == str:
                    this_val = exp_dict[key]
                    this_val_sub = os.path.expandvars(this_val)
                    if this_val != this_val_sub:
                        logger.info("env substitution for {}: {} with {}".format(
                            k, this_val, this_val_sub))
                    print(this_val, this_val_sub)
                    exp_dict[key] = this_val_sub      
                """

    # insert defaults
    # print("defaults=", defaults)
    for k, v in defaults.items():
        # If the key is not in exp_dict, then we will insert it
        # by default. Otherwise, if it exists then validate it
        # (unless it's also a dict, then recurse into it)
        if k not in exp_dict:
            logger.info(
                "Inserting default arg into exp dict: {}={}".format(k, v.default)
            )
            # print(" exp dict is:", exp_dict)
            exp_dict[k] = v.default

            # sanity check, the default value should match the allowable types
            defaults[k].validate(v.default)
        else:
            # validate the arg
            if type(v) is not dict:
                # print(k, v)
                defaults[k].validate(exp_dict[k])

class AugmentedIterable(IterableDataset):
    def __init__(self, loader, model, mode):
        """
        Return an iterable loader conditioned on the given loader.
            Depending on the selected mode, this will return an 'augmented'
            form of the original loader where each minibatch will be either
            the reconstructed version, the mixed version (using the k axis),
            or simply return its original input.

        n_passes: how many passes to do over the original loader.
            For instance, if `n_passes=5`, then the augmented loader
            will have a length 5x as much as the original.

        """
        super(AugmentedIterable).__init__()
        SUPPORTED_MODES = ["gt", "recon", "mixup"]
        if mode not in SUPPORTED_MODES:
            raise Exception("mode must be supported")
        self.loader = loader
        self.dataset = self.loader.dataset
        self.model = model
        self.mode = mode

    def __iter__(self):

        self.model.eval()

        for batch in self.loader:
            images = batch["images"]
            labels = batch["labels"]
            if self.mode == "recon":
                yield {"images": self.model.reconstruct(images), "labels": labels}
            elif self.mode == "mixup":
                mix_imgs, _ = self.model.generate_on_batch(images)
                yield {"images": mix_imgs.transpose(0, 1), "labels": labels}
            else:
                yield {"images": images, "labels": labels}

    def __len__(self):
        return len(self.loader)


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def count_params(module, trainable_only=True):
    """Count the number of parameters in a
    module.
    :param module: PyTorch module
    :param trainable_only: only count trainable
      parameters.
    :returns: number of parameters
    :rtype:
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num


def check_params_for_nans(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise Exception("{} has nans in grad buffer".format(name))
