import os
import torch
import sys
from torch import nn
import numpy as np
import time
from src import datasets
from haven import haven_wizard as hw
import json

from src.metrics import models_clf
from src.fid import fid_score
from src.models import InfoGAN
from src.models.utils import (Argument,
                              FidWrapper,
                              validate_and_insert_defaults,
                              precompute_fid_stats,
                              load_json_from_file)
from src.models.haven_utils import get_checkpoint, save_checkpoint

# Parallel stuff
import torch.multiprocessing as mp
import torch.distributed as dist

from src import setup_logger
logger = setup_logger.get_logger(__name__)

def setup_mpi(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

NONETYPE = type(None)
DEFAULTS = {
    "load_strict": Argument('load_strict', True, [bool]),
    "epochs": Argument('epochs', 200, [int]),
    "finetune": Argument('finetune', False, [bool]),
    "save_every": Argument('save_every', None, [int, NONETYPE]),
    "fid_every": Argument('fid_every', 5, [int]), # compute fid every this many epochs
    "fid_N": Argument('fid_N', 5000, [int]), # how many real/fake samples for FID computation
    "fid_pretrained": Argument('fid_pretrained', None, [str, NONETYPE]), # fid feature extractor
    "semi_sup": Argument('semi_sup', False, [bool]),
    "pretrained": Argument('pretrained', None, [str, NONETYPE]),
    "dataset": {
        "name": Argument("name", "emnist", [str]),
        "k_shot": Argument("k_shot", 5, [int]),
        "seed": Argument("seed", 0, [int]),
        "input_size": Argument("input_size", 32, [int]),
        "pad_length": Argument("pad_length", 0, [int]),
        "transform_kwargs": Argument("transform_kwargs", {}, [dict, NONETYPE])
    },
    "batch_size": Argument('batch_size', 32, [int]),
    "model": Argument("model", {}, [dict]), # model kwargs, see infogan.py
    "optim": {
        "beta1": Argument("beta1", 0.9, [float]),
        "beta2": Argument("beta2", 0.9, [float]),
        "lr": Argument("lr", 2e-4, [float]),
        "n_gen": Argument("n_gen", 1, [int]),
        "weight_decay": Argument("weight_decay", 1e-6, [float]),
        "eps": Argument("eps", 1e-8, [float])
    },
    # args that are no longer used, added as dummy args to stop
    # script from crashing:
    "pretrained_enc": Argument("pretrained_enc", None, [NONETYPE])
}

def validate_args(dd):
    if not dd["finetune"] and dd["dataset"]["pad_length"] > 0:
        raise Exception("dataset.pad_length only supported for finetuning")
    if dd["semi_sup"] and not dd["finetune"]:
        raise Exception("semi-sup mode only supported if finetune==True")

def trainval(exp_dict, savedir, args):
    logger.info("Validating and inserting defaults...")
    validate_and_insert_defaults(exp_dict, DEFAULTS)
    logger.info("Extra validating args...")
    validate_args(exp_dict)

    if not os.path.exists("{}/exp_dict.json".format(savedir)):
        # This would trigger if we're launching an experiment
        # using launch.py.
        with open("{}/exp_dict.json".format(savedir), "w") as f:
            f.write(json.dumps(exp_dict))
    else:
        # If we are using `launch_haven.py`, it does write its own `exp_dict.json`
        # into `savedir` but that is before the arg validation, and here we want
        # `exp_dict.json` to inherit the defaults from `DEFAULTS`. Because of
        # this, launch_haven.py has a specific arg called "--disable_rewrite"
        # which should be set to true if an experiment is being resumed.
        pass

    world_size = int(os.environ["WORLD_SIZE"])
    if world_size == 0:
        logger.info("WORLD_SIZE==0, running on a single process")
        _trainval(rank=0, world_size=1, exp_dict=exp_dict, savedir=savedir, args=args)
    else:
        logger.info("WORLD_SIZE>0, running on multiprocess...")
        mp.spawn(
            _trainval,
            args=(world_size, exp_dict, savedir, args),
            nprocs=world_size,
            join=True,
        )

def _trainval(rank, world_size, exp_dict, savedir, args):

    logger.info("RANK: {}, WORLD SIZE: {}".format(rank, world_size))
    setup_mpi(rank, world_size)

    dataset_name = exp_dict["dataset"]["name"]
    dataset_input_size = exp_dict["dataset"]["input_size"]
    dataset_seed = exp_dict["dataset"]["seed"]
    dataset_transform = exp_dict["dataset"]["transform_kwargs"]

    batch_size = exp_dict["batch_size"]

    load_strict = True
    if "load_strict" in exp_dict:
        load_strict = exp_dict["load_strict"]
    if not load_strict:
        logger.warning(
            "`load_strict` is not set, which means loading pre-trained weights "
            " may work even when the model definition has changed"
        )

    # Load datasets
    # -------------

    train_dataset = datasets.get_dataset(
        class_split="train",
        datadir=args.datadir,
        dataset=dataset_name,
        seed=dataset_seed,
        input_size=dataset_input_size,
        which_set=None,
        k_shot=None,
        return_pairs=True,
        transform_kwargs=dataset_transform,
    )
    
    distributed = True if world_size > 0 else False
    train_loader, train_sampler = datasets.get_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        distributed=distributed,
    )
    logger.info("train dataset: {}\n".format(train_dataset))
    dev_sampler = None

    # Load model and get checkpoint
    model = InfoGAN(
        input_size=dataset_input_size,
        exp_dict=exp_dict,
        n_classes=train_loader.dataset.n_classes,
        train=True,
        rank=rank,
        finetune=exp_dict["finetune"],
        verbose=True,
    )

    if exp_dict["fid_pretrained"] is not None:
        fid_model = models_clf.get_model(
            n_classes=train_dataset.n_classes,
            freeze_all_except=None,
        )
        clf_state_dict = torch.load(exp_dict["fid_pretrained"])
        clf_cfg = load_json_from_file(
            "{}/exp_dict.json".format(os.path.dirname(exp_dict["fid_pretrained"]))
        )
        # Verify that the k_shot here is the same as what is
        # specified in the pretrained exp_dict.json.
        if exp_dict["dataset"]["seed"] != clf_cfg["dataset"]["seed"]:
            raise Exception(
                "dataset.seed does not match that of FID pretrained dataset.seed"
            )
        logger.info(
            "fid_pretrained is set, so using classifier features: {}".format(
                exp_dict["fid_pretrained"]
            )
        )
        fid_model.set_state_dict(clf_state_dict)
        fid_model = FidWrapper(fid_model.model.f)
    else:
        fid_model = None

    fid_every = exp_dict["fid_every"]
    unsup_loader = None
    if fid_every >= 0:
        fid_N = exp_dict["fid_N"]
        train_fid_mean, train_fid_sd = precompute_fid_stats(
            train_loader, batch_size, fid_N, model=fid_model
        )

        if exp_dict["finetune"]:
            dev_dataset = datasets.get_dataset(
                class_split="valid",
                datadir=args.datadir,
                dataset=dataset_name,
                seed=dataset_seed,
                input_size=dataset_input_size,
                k_shot=exp_dict["dataset"]["k_shot"],
                which_set="supports",
                return_pairs=True,
                transform_kwargs=dataset_transform,
            )
            logger.info("dev dataset: {}\n".format(dev_dataset))
            dataset_pad_M = exp_dict["dataset"]["pad_length"]
            if dataset_pad_M > 0:
                dev_dataset = datasets.DuplicateDatasetMTimes(
                    dev_dataset, M=dataset_pad_M
                )
            dev_loader, dev_sampler = datasets.get_loader(
                dev_dataset,
                batch_size=batch_size,
                num_workers=args.num_workers,
                distributed=distributed,
            )
            
            valid_dataset = datasets.get_dataset(
                class_split="valid",
                datadir=args.datadir,
                dataset=dataset_name,
                seed=dataset_seed,
                input_size=dataset_input_size,
                k_shot=exp_dict["dataset"]["k_shot"],
                which_set="valid",
                return_pairs=True,
                transform_kwargs=dataset_transform,
            )
            logger.info("valid dataset: {}".format(valid_dataset))
            valid_loader, _ = datasets.get_loader(
                valid_dataset,
                batch_size=batch_size,
                num_workers=args.num_workers,
                distributed=distributed,
            )

            logger.info("Precomputing global FID stats for valid set")
            valid_fid_mean, valid_fid_sd = precompute_fid_stats(
                valid_loader, batch_size, fid_N, model=fid_model
            )

            #logger.info("Precomputing FID stats per class for valid set")
            #valid_class_to_stats = _precompute_fid_stats_per_class(
            #    valid_dataset, batch_size, fid_N, model=fid_model
            #)

            if exp_dict["semi_sup"]:
                unsup_loader, _ = datasets.get_loader(
                    valid_dataset,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                    distributed=distributed,
                )

    # Explicitly set what gpu to put the weights on.
    # If map_location is not set, each rank (gpu) will
    # load these onto presumably gpu0, causing an OOM
    # if we run this code under a resuming script.
    chk_dict = get_checkpoint(
        savedir,
        return_model_state_dict=True,
        map_location=lambda storage, loc: storage.cuda(rank),
    )
    if exp_dict["pretrained"] is not None:
        pretrained_chkpt = torch.load(exp_dict["pretrained"])
        pretrained_cfg = load_json_from_file(
            "{}/exp_dict.json".format(os.path.dirname(exp_dict["pretrained"]))
        )
        logger.info("Loading pretrained model: {}".format(exp_dict["pretrained"]))
        model.set_state_dict(
            pretrained_chkpt,
            load_opt=False if exp_dict["finetune"] else True,
            strict=True,
        )
        # Verify that the k_shot here is the same as what is
        # specified in the pretrained exp_dict.json.
        if exp_dict["dataset"]["seed"] != pretrained_cfg["dataset"]["seed"]:
            raise Exception(
                "dataset.seed does not match that of pretrained dataset.seed"
            )

    if len(chk_dict["model_state_dict"]):
        model.set_state_dict(chk_dict["model_state_dict"], strict=load_strict)

    # Run Train-Val loop
    # -----------------------------
    max_epochs = exp_dict["epochs"]
    save_every = exp_dict["save_every"]
    if exp_dict["finetune"]:
        # If we're just doing supervised finetuning, use global FID
        # otherwise, use averaged per-class FID
        if exp_dict["semi_sup"]:
            # aoc = average over classes
            chk_metric = "fid_valid" # used to be cfid_Valid but too expensive for og
        else:
            chk_metric = "fid_valid"
    else:
        chk_metric = "fid"

    logger.info("chk_metric: {}".format(chk_metric))

    if len(chk_dict["score_list"]) == 0:
        best_metric = np.inf
    else:
        metric_scores = [
            score[chk_metric] for score in chk_dict["score_list"] if chk_metric in score
        ]
        if len(metric_scores) == 0:
            best_metric = np.inf
        else:
            best_metric = min(metric_scores)

    if args.dry_run:
        sys.stderr.write("dry run set, terminating...\n")
        return

    logger.info("Starting epoch: {}".format(chk_dict["epoch"]))
    for epoch in range(chk_dict["epoch"], max_epochs):

        t0 = time.time()

        # TODO: reduce cpu stats as well???
        if rank == 0:
            score_dict = {}
            score_dict["epoch"] = epoch

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if dev_sampler is not None:
            dev_sampler.set_epoch(epoch)

        # (1) Train GAN.
        train_dict_ = model.train_on_loader(
            train_loader if not exp_dict["finetune"] else dev_loader,
            unsup_loader,
            epoch=epoch,
            savedir=savedir,
            pbar=world_size <= 1,
        )
        train_dict = {("train_" + key): val for key, val in train_dict_.items()}

        if rank == 0:
            # TODO: currently we don't do barriers, we only
            # save the metrics that are on gpu0

            score_dict.update(train_dict)
            # score_dict.update(valid_dict)

            score_dict["time"] = time.time() - t0

            if fid_every >= 0 and epoch % fid_every == 0 and epoch > 0:
                logger.info("Computing FID between train and generated...")
                generated_imgs_train = model.sample_from_loader(train_loader, N=fid_N)
                this_fid_train = fid_score.calculate_fid_given_imgs_and_stats(
                    generated_imgs_train,
                    train_fid_mean,
                    train_fid_sd,
                    batch_size,
                    device=0,
                    model=fid_model,
                )
                score_dict["fid"] = this_fid_train

                if exp_dict["finetune"]:
                    logger.info("Computing FID between generated and valid")
                    generated_imgs_dev = model.sample_from_loader(valid_loader, N=fid_N)
                    this_fid_valid = fid_score.calculate_fid_given_imgs_and_stats(
                        generated_imgs_dev,
                        valid_fid_mean,
                        valid_fid_sd,
                        batch_size,
                        device=0,
                        model=fid_model,
                    )
                    score_dict["fid_valid"] = this_fid_valid

                    """
                    logger.info(
                        "Computing avg per-class FID between generated and valid"
                    )
                    avg_per_class_fids = []
                    for key in valid_class_to_stats.keys():
                        generated_imgs_dev_this_class = model.sample_from_loader(
                            dev_loader, N=fid_N, label=key
                        )
                        this_fidpc_valid = fid_score.calculate_fid_given_imgs_and_stats(
                            generated_imgs_dev_this_class,
                            valid_class_to_stats[key][0],
                            valid_class_to_stats[key][1],
                            batch_size,
                            device=0,
                            model=fid_model,
                        )
                        avg_per_class_fids.append(this_fidpc_valid)
                    logger.info("avg_per_class_fids:" + str(avg_per_class_fids))
                    score_dict["cfid_valid"] = np.mean(avg_per_class_fids)
                    score_dict["cfid_valid_sd"] = np.std(avg_per_class_fids)
                    """

                if score_dict[chk_metric] < best_metric:
                    logger.info(
                        "Best metric: from {}={} to {}={}".format(
                            chk_metric, best_metric, chk_metric, score_dict[chk_metric]
                        )
                    )
                    best_metric = score_dict[chk_metric]
                    save_checkpoint(
                        savedir,
                        fname_suffix="." + chk_metric,
                        score_list=chk_dict["score_list"],
                        model_state_dict=model.get_state_dict(),
                        verbose=False,
                    )

            chk_dict["score_list"] += [score_dict]

            # Save checkpoint
            save_checkpoint(
                savedir,
                score_list=chk_dict["score_list"],
                model_state_dict=model.get_state_dict(),
                verbose=False,
            )

            # If `save_every` is defined, save every
            # this many epochs.
            if save_every is not None:
                if epoch > 0 and epoch % save_every == 0:
                    save_checkpoint(
                        savedir,
                        fname_suffix="." + str(epoch),
                        score_list=chk_dict["score_list"],
                        model_state_dict=model.get_state_dict(),
                        verbose=False,
                    )

    print("Experiment completed")