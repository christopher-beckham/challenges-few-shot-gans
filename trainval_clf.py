import os
import numpy as np
import torch
import time
import sys
from torch.utils.data import Dataset, DataLoader, Subset

from haven import haven_utils as hu
from haven import haven_wizard as hw

from src import datasets
from src.metrics import models_clf

from src.models.utils import validate_and_insert_defaults, Argument
from src.models.haven_utils import save_checkpoint, save_pkl, get_checkpoint

from src import setup_logger

logger = setup_logger.get_logger(__name__)


def get_train_trainval_split(train_dataset, train_percent, split_seed):

    # Cut up full training set into its own
    # actual train set and valid set.
    idcs = np.arange(0, len(train_dataset))
    logger.info("split seed: {}".format(split_seed))
    rnd_state = np.random.RandomState(split_seed)
    rnd_state.shuffle(idcs)

    # Set aside 90% for train and 10%
    # for valid.
    logger.info("Train percentage: {}".format(train_percent))
    train_idcs = idcs[0 : int(len(idcs) * train_percent)]
    valid_idcs = idcs[int(len(idcs) * train_percent) : :]
    logger.info("Len of train set: {}".format(len(train_idcs)))
    logger.info("Len of train-valid set: {}".format(len(valid_idcs)))

    train_actual = Subset(train_dataset, indices=train_idcs)
    valid_actual = Subset(train_dataset, indices=valid_idcs)

    return train_actual, valid_actual


NONETYPE = type(None)
DEFAULTS = {
    "dataset": {
        # actual dataset seed
        "name": Argument("name", "emnist_fs", [str]),
        "input_size": Argument("input_size", 32, [int]),
        "seed": Argument("seed", 42, [int]),
        "train_percentage": Argument("train_percentage", 0.9, [float]),
        "clf_transform_kwargs": Argument("clf_transform_kwargs", {}, [dict, NONETYPE]),
        # not used anymore
        "n_channels": Argument("n_channels", 1, [int]),
    },
    "epochs": Argument("epochs", 200, [int]),
    "batch_size": Argument("batch_size", 64, [int]),
    "optim": {
        "lr": Argument("lr", 2e-4, [float]),
        "beta1": Argument("beta1", 0.9, [float]),
        "beta2": Argument("beta2", 0.999, [float]),
        "weight_decay": Argument("weight_decay", 0.0, [float]),
    },
}


def trainval(exp_dict, savedir, args):

    validate_and_insert_defaults(exp_dict, DEFAULTS)

    dataset_seed = exp_dict["dataset"]["seed"]
    train_percent = exp_dict["dataset"]["train_percentage"]
    epochs = exp_dict["epochs"]
    # mixup_alpha = exp_dict.get("mixup_alpha", None)

    # We train the AE on this.
    train_dataset = datasets.get_dataset(
        class_split="train",
        k_shot=None,
        datadir=args.datadir,
        dataset=exp_dict["dataset"]["name"],
        input_size=exp_dict["dataset"]["input_size"],
        seed=dataset_seed,
        transform_kwargs=exp_dict["dataset"]["clf_transform_kwargs"],
    )
    logger.info("\n" + str(train_dataset))

    train_actual, valid_actual = get_train_trainval_split(
        train_dataset, train_percent=train_percent, split_seed=0
    )

    train_loader = DataLoader(
        train_actual,
        shuffle=True,
        batch_size=exp_dict["batch_size"],
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_actual,
        shuffle=True,
        batch_size=exp_dict["batch_size"],
        num_workers=args.num_workers,
    )

    # Pretrain classifier
    # -------------------
    optim_args = exp_dict["optim"]
    model_clf = models_clf.get_model(
        n_classes=train_dataset.n_classes,
        lr=optim_args["lr"],
        beta1=optim_args["beta1"],
        beta2=optim_args["beta2"],
        weight_decay=optim_args["weight_decay"],
        freeze_all_except=None,
    )
    chk_dict = get_checkpoint(savedir, return_model_state_dict=True)

    if len(chk_dict["model_state_dict"]):
        model_clf.set_state_dict(chk_dict["model_state_dict"])

    val_acc_best = max(
        [s.get("val_score", -np.inf) for s in chk_dict["score_list"]] + [-np.inf]
    )

    if args.dry_run:
        sys.stderr.write("dry run set, terminating...\n")
        return

    for epoch in range(chk_dict["epoch"], epochs):

        t0 = time.time()

        score_dict = {}
        score_dict["epoch"] = epoch

        # Train model.
        # Since `savedir` is defined, after every epoch it will invoke
        # `vis_on_batch` to generate images.
        train_dict = model_clf.train_on_loader(train_loader, savedir=savedir)

        score_dict.update({("train_" + k): v for k, v in train_dict.items()})

        valid_dict = model_clf.val_on_loader(valid_loader)
        score_dict.update({("valid_" + k): v for k, v in valid_dict.items()})

        score_dict["time"] = time.time() - t0

        score_dict["val_score"] = score_dict["valid_acc"]

        chk_dict["score_list"] += [score_dict]

        # Save best model if 'val_acc' improves

        if score_dict.get("val_score", -np.inf) >= val_acc_best:
            logger.info(
                "new best validation acc: from {} to {}".format(
                    val_acc_best, score_dict["val_score"]
                )
            )
            val_acc_best = score_dict["val_score"]
            score_dict["best"] = True
            save_checkpoint(
                savedir,
                fname_suffix="_best",
                score_list=chk_dict["score_list"],
                model_state_dict=model_clf.get_state_dict(),
                verbose=True,
            )
        # Save last chkpt
        save_checkpoint(
            savedir,
            score_list=chk_dict["score_list"],
            model_state_dict=model_clf.get_state_dict(),
            verbose=False,
        )

    print("Experiment completed et epoch %d" % epoch)
