"""
Helper functions extracted from haven-ai:
https://github.com/haven-ai/haven-ai
"""

import os
import pickle
import torch

def torch_save(fname, obj):
    """Save data in torch format.

    Parameters
    ----------
    fname : str
        File name
    obj : [type]
        Data to save
    """
    # Create folder
    os.makedirs(os.path.dirname(fname), exist_ok=True)  # TODO: add makedirs parameter?

    # Define names of temporal files
    fname_tmp = fname + ".tmp"  # TODO: Make the safe flag?

    torch.save(obj, fname_tmp)
    if os.path.exists(fname):
        os.remove(fname)
    os.rename(fname_tmp, fname)


def get_checkpoint(savedir, return_model_state_dict=False, map_location=None):
    chk_dict = {}

    # score list
    score_list_fname = os.path.join(savedir, "score_list.pkl")
    if os.path.exists(score_list_fname):
        score_list = hu.load_pkl(score_list_fname)
    else:
        score_list = []

    chk_dict["score_list"] = score_list
    if len(score_list) == 0:
        chk_dict["epoch"] = 0
    else:
        chk_dict["epoch"] = score_list[-1]["epoch"] + 1

    model_state_dict_fname = os.path.join(savedir, "model.pth")
    if return_model_state_dict:
        if os.path.exists(model_state_dict_fname):
            chk_dict["model_state_dict"] = hu.torch_load(
                model_state_dict_fname, map_location=map_location
            )

        else:
            chk_dict["model_state_dict"] = {}

    return chk_dict

def save_checkpoint(
    savedir, score_list, model_state_dict=None, images=None, images_fname=None, fname_suffix="", verbose=True
):

    # save score_list
    score_list_fname = os.path.join(savedir, "score_list%s.pkl" % fname_suffix)
    save_pkl(score_list_fname, score_list)
    # if verbose:
    # print('> Saved "score_list" as %s' %
    #       os.path.split(score_list_fname)[-1])

    # save model
    if model_state_dict is not None:
        model_state_dict_fname = os.path.join(savedir, "model%s.pth" % fname_suffix)
        torch_save(model_state_dict_fname, model_state_dict)
        # if verbose:
        # print('> Saved "model_state_dict" as %s' %
        #       os.path.split(model_state_dict_fname)[-1])

def save_pkl(fname, data, with_rename=True, makedirs=True):
    """Save data in pkl format.

    Parameters
    ----------
    fname : str
        File name
    data : [type]
        Data to save in the file
    with_rename : bool, optional
        [description], by default True
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    # Create folder
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)

    # Save file
    if with_rename:
        fname_tmp = fname + "_tmp.pth"
        with open(fname_tmp, "wb") as f:
            pickle.dump(data, f)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(fname_tmp, fname)
    else:
        with open(fname, "wb") as f:
            pickle.dump(data, f)