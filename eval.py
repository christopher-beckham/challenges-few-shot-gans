import argparse
import json
import os
import pickle
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from src import datasets
from src.metrics import models_clf

from src import setup_logger

logger = setup_logger.get_logger(__name__)

from src.models import utils_viz
from src.models import InfoGAN

from src.models.utils import FidWrapper, precompute_fid_activations, load_json_from_file

import torch.distributed as dist

import prdc


def setup_mpi(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def run(exp_dir, savedir, seed, args):

    # This is needed if we want to call g_model
    # in ae_mixup.py, because it is wrapped in
    # that DDP container.
    setup_mpi(rank=0, world_size=1)

    if seed is not None:
        logger.info("Setting seed to: {}".format(seed))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Setup.
    logger.info("Loading {}...".format(exp_dir))
    exp_dict = load_json_from_file("{}/exp_dict.json".format(exp_dir))

    dataset_name = exp_dict["dataset"]["name"]
    dataset_input_size = exp_dict["dataset"]["input_size"]
    dataset_seed = exp_dict["dataset"]["seed"]

    if "k_shot" in exp_dict["dataset"]:
        dataset_k_shot = exp_dict["dataset"]["k_shot"]
    elif "k_shot" in exp_dict["dataset"]:
        logger.info(
            "`dataset.k_shot` not found but detected in ae_exp_dict, "
            + "so using this..."
        )
        dataset_k_shot = exp_dict["dataset"]["k_shot"]
    else:
        raise Exception(
            "`dataset.k_shot` not found in either the current "
            + "exp_dict or that of ae_exp_dict"
        )

    # datadir =
    num_workers = 4

    train_dataset = datasets.get_dataset(
        class_split="train",
        datadir=args.datadir,
        dataset=dataset_name,
        seed=dataset_seed,
        input_size=dataset_input_size,
        k_shot=None,
    )

    dev_dataset = datasets.get_dataset(
        class_split="valid",
        datadir=args.datadir,
        dataset=dataset_name,
        seed=dataset_seed,
        input_size=dataset_input_size,
        k_shot=dataset_k_shot,
        which_set="supports",
    )

    train_loader, _ = datasets.get_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        distributed=False,
    )

    dev_loader, _ = datasets.get_loader(
        dev_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        distributed=False,
    )

    if exp_dict["fid_pretrained"] is not None:
        logger.info("Loading pre-trained classifier...")
        fid_model = models_clf.get_model(
            n_classes=train_dataset.n_classes, freeze_all_except=None,
        )
        clf_state_dict = torch.load(exp_dict["fid_pretrained"])
        fid_model.set_state_dict(clf_state_dict)
        fid_model = FidWrapper(fid_model.model.f)
    else:
        fid_model = None

    # Load the pre-trained autoencoder.
    # ---------------------------------

    if args.cpu:
        logger.info("cpu mode set...")
        rank = "cpu"
    else:
        rank = 0
    model = InfoGAN(
        exp_dict=exp_dict,
        n_classes=train_dataset.n_classes,
        input_size=dataset_input_size,
        # no need to print AE stuff here
        verbose=False,
        rank=rank,
        train=True,
    )

    if args.disable_ema:
        logger.info("Disabling EMA mode")
        model.use_ema = False

    logger.info(
        "Loading model state dict: {}".format(os.path.join(exp_dir, args.model))
    )
    model_state_dict = torch.load(os.path.join(exp_dir, args.model))
    logger.info("Loading AE {} ...".format(exp_dir))
    model.set_state_dict(model_state_dict, load_opt=False)

    exp_dir = os.path.join(exp_dir, "")  # add trailing slash

    # -------------------------------------
    # Generate precision and recall metrics
    # -------------------------------------

    if args.prdc:

        valid_dataset = datasets.get_dataset(
           class_split="valid",
           datadir=args.datadir,
           dataset=dataset_name,
           seed=dataset_seed,
           input_size=dataset_input_size,
           k_shot=dataset_k_shot,
           which_set="valid",
        )

        valid_loader, _ = datasets.get_loader(
           valid_dataset,
           batch_size=args.batch_size,
           num_workers=num_workers,
           distributed=False,
        )
        
        valid_acts = precompute_fid_activations(
            valid_loader, batch_size=32, N=None, model=fid_model
        )
        gen_dataset = model.sample_from_loader(valid_loader, verbose=False)
        gen_loader = DataLoader(gen_dataset, batch_size=32)
        gen_acts = precompute_fid_activations(
            gen_loader, batch_size=32, N=None, model=fid_model, extract_fn=lambda x: x
        )
        logger.info("valid_acts shape: {}".format(valid_acts.shape))
        logger.info("gen_acts shape: {}".format(gen_acts.shape))
        prdc_results = prdc.compute_prdc(
            real_features=valid_acts, fake_features=gen_acts, nearest_k=3
        )
        prdc_outfile = "{}/prdc.pkl".format(exp_dir)
        logger.info("Results: {}".format(prdc_results))
        logger.info("Saving precision-recall metrics to {}".format(prdc_outfile))
        with open(prdc_outfile, "wb") as f:
            pickle.dump(prdc_results, f)

    # -----------------
    # Generating images
    # -----------------

    actual_savedir = "{}/{}".format(savedir, "/".join(exp_dir.split("/")[-3:]))
    if not os.path.exists(actual_savedir):
        os.makedirs(actual_savedir)
    logger.info("Save dir: {}".format(actual_savedir))

    logger.info("Generating images...")

    train_batch = iter(train_loader).next()
    dev_batch = iter(dev_loader).next()
    # valid_batch = iter(valid_loader).next()
    model.vis_on_batch(
        train_batch, savedir=actual_savedir, split="train", dataset=train_loader.dataset
    )
    model.vis_on_batch(
        dev_batch, savedir=actual_savedir, split="dev", dataset=dev_loader.dataset
    )

    # model.vis_on_batch(train_batch, savedir=actual_savedir, split="train_dev", batch2=dev_batch)


def reconstruct_for_all_labels(
    model,
    train_batch,
    train_labels_unique,
    dev_batch,
    dev_labels_unique,
    valid_batch,
    savedir,
):
    train_acc = utils_viz.reconstruct_for_all_labels(
        model,
        train_batch["images"],
        train_batch["labels"][:, 0],
        train_labels_unique,
        "{}/images/train_recon_all_labels.png".format(savedir),
    )
    logger.info("Training acc: {}".format(train_acc))

    dev_acc = utils_viz.reconstruct_for_all_labels(
        model,
        dev_batch["images"],
        dev_batch["labels"][:, 0],
        dev_labels_unique,
        "{}/images/dev_recon_all_labels.png".format(savedir),
    )
    logger.info("Dev acc: {}".format(dev_acc))

    valid_acc = utils_viz.reconstruct_for_all_labels(
        model,
        valid_batch["images"],
        valid_batch["labels"][:, 0],
        dev_labels_unique,
        "{}/images/valid_recon_all_labels.png".format(savedir),
    )
    logger.info("Valid acc: {}".format(valid_acc))


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description="TODO")

        parser.add_argument(
            "--experiment",
            type=str,
            default=None,
            required=True,
            help="Path to the experiment",
        )

        parser.add_argument(
            "--savedir", type=str, required=True
        )
        parser.add_argument("--datadir", type=str, default="/mnt/public/datasets")
        parser.add_argument("--model", type=str, default="model.pth")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--disable_ema", action="store_true")
        parser.add_argument("--cpu", action="store_true")
        parser.add_argument("--prdc", action="store_true")
        args = parser.parse_args()
        return args

    args = parse_args()

    run(args.experiment, args.savedir, args.seed, args)
