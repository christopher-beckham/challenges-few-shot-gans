import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchvision.utils import save_image, make_grid

import os
import json

import argparse

# from src.metrics import svm_acc
from haven import haven_wizard as hw

from src.models import InfoGAN
from src.datasets import utils as dataset_utils
from src import datasets

from src.metrics import models_clf
from src.datasets import DuplicateDatasetMTimes

from src.models import utils_viz
from src.models.utils import (validate_and_insert_defaults,
                              get_checkpoint,
                              Argument)

from src import setup_logger
logger = setup_logger.get_logger(__name__)

NONETYPE = type(None)
DEFAULTS = {
    "mode": Argument("mode", "baseline", [str], ["baseline", "sample"]),
    "epochs": Argument("epochs", 1000, [int]),
    "bn_moving_avgs": Argument("bn_moving_avgs", False, [bool]),
    "freeze_all_except": Argument("freeze_all_except", [], [list]),  # TODO: deprecate??
    "pretrained": Argument("pretrained", None, [list, tuple]),  # 2-tuple in format (clf_exp, ae_exp)
    "n_samples_per_class": Argument("n_samples_per_class", 0, [int]),  # number of augmented samples per class
    "dataset": {
        "k_shot": Argument("k_shot", None, [NONETYPE, int]),  # k_shot
        "transform_kwargs": Argument("transform_kwargs", None, [NONETYPE, dict]),
        "pad_length": Argument("pad_length", 0, [int])
    },
    #'dataset': None,
    "finetune": {
        "transform_kwargs": Argument("transform_kwargs", None, [NONETYPE, dict])  # kwargs for transform_clf
    },
    "batch_size": Argument("batch_size", 32, [int]),  # batch size for training
    "aug_batch_size": Argument("aug_batch_size", 32, [int]),  # batch size for generating new images
    "N_fakeval": Argument("N_fakeval", 1000, [int]), # number of samples per class for fake valid set
    "use_mixup": Argument("use_mixup", False, [bool]),  # enable mixup mode for classifier
    "stdev": Argument("stdev", 1.0, [float]),  # stdev of prior distn
    "mixup": {
        "dist": Argument("dist", "uniform", [str], ["uniform", "beta"]),  # do we use beta or uniform
        "alpha": Argument("alpha", 1.0, [float]),  # U(0,alpha) for uniform, beta(alpha, alpha) for beta
        "mix_labels": Argument("mix_labels", True, [bool])  # (input mixup only) mix labels as well?
    },
    "ignore_dev_set": Argument("ignore_dev_set", False, [bool]),  # if mode==augment, do not use original supports
    "optim": {
        "lr": Argument("lr", 2e-4, [float]),
        "beta1": Argument("beta1", 0.9, [float]),
        "beta2": Argument("beta2", 0.999, [float]),
        "weight_decay": Argument("weight_decay", 0.0, [float]),
        "eps": Argument("eps", 1e-8, [float]),
        # TODO
        "step_size": Argument("step_size", None, [NONETYPE]),
        "gamma": Argument("gamma", None, [NONETYPE])
    },
    "save_every": Argument("save_every", 1, [int]),  # save model/metrics every this many epochs
    "validate_every": Argument("validate_every", 100, [int]),
    # just a dummy argument to make haven shut up if I want to do repeat runs
    "rnd": Argument("rnd", None, [int, NONETYPE]),
    "ignore_k_shot_warning": Argument("ignore_k_shot_warning", False, [bool])
}

def validate_args(dd):
    if dd["n_samples_per_class"] > 0 and dd["mode"] == "baseline":
        raise Exception("n_samples_per_class>0 can only be for non-baselines")

def viz_real_and_augmented_images(
    real_batch, aug_batch, outfile, invert=False, class_to_metric=None
):
    real_imgs = real_batch["images_aug"]
    real_labels = real_batch["labels"]
    unique_labels = real_labels.unique()
    aug_imgs = aug_batch["images_aug"]
    aug_labels = aug_batch["labels"]

    # assert len(aug_imgs) == len(metric_batch)

    # Each element ('row') in buf is for a specific class
    buf = []
    text = []
    for label in unique_labels:
        label = label.item()
        real_imgs_for_label = real_imgs[real_labels == label]
        aug_imgs_for_label = aug_imgs[aug_labels == label]
        #print(len(real_imgs_for_label), len(aug_imgs_for_label))
        if class_to_metric is not None:
            metric_for_label = class_to_metric[label]
        # Turn them into a single 'row' with a black img in
        # between.
        this_row = torch.cat(
            (real_imgs_for_label, real_imgs_for_label[0:1] * 0 - 1, aug_imgs_for_label),
            dim=0,
        )
        buf.append(this_row)
        # metrics_mean = "%.2f" % metrics_for_label.mean()
        # metrics_sd = "%.2f" % metrics_for_label.std()
        # text.append("p(c) = {} +/- {}".format(metrics_mean, metrics_sd))
        if class_to_metric is not None:
            text.append("metric={}".format(metric_for_label))
        else:
            text.append("")
    total_buf = torch.cat(buf, dim=0) * 0.5 + 0.5
    if invert:
        total_buf = 1 - total_buf
    prefinal_img = make_grid(total_buf, nrow=buf[0].size(0), pad_value=1)
    final_img = utils_viz._annotate_image(
        prefinal_img, im_size=32, pad_size=2, labels=text  # TODO,
    )
    final_img.save(outfile)


def viz_images(real_batch, outfile, invert=False):
    real_imgs = real_batch["images"]
    real_labels = real_batch["labels"]
    unique_labels = real_labels.unique()
    # Each element ('row') in buf is for a specific class
    buf = []
    for label in unique_labels:
        label = label.item()
        this_row = real_imgs[real_labels == label]
        buf.append(this_row)
    total_buf = torch.cat(buf, dim=0) * 0.5 + 0.5
    if invert:
        total_buf = 1 - total_buf
    save_image(total_buf, outfile, nrow=buf[0].size(0), pad_value=1)


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """

    validate_and_insert_defaults(exp_dict, DEFAULTS)
    validate_args(exp_dict)

    # Load exp dictionaries for AE and classifier
    # -------------------------------------------

    METADATA = {}

    # You can either specify ae_exp / clf_exp individually,
    # or specify them both in a tuple as 'ae_and_clf_exp'.
    # The latter is to support REDACTED's json style of
    # launching experiments.
    clf_exp_model_path = exp_dict["pretrained"][0]
    ae_exp_model_path = exp_dict["pretrained"][1]
    clf_exp_dir = os.path.dirname(clf_exp_model_path)
    ae_exp_dir = os.path.dirname(ae_exp_model_path)

    ae_exp_dict = json.loads(open("{}/exp_dict.json".format(ae_exp_dir)).read())
    clf_exp_dict = json.loads(open("{}/exp_dict.json".format(clf_exp_dir)).read())
    dataset_name = clf_exp_dict["dataset"]["name"]
    dataset_input_size = clf_exp_dict["dataset"]["input_size"]

    # This is checked against the clf_exp_dict
    # seed.
    dataset_seed = clf_exp_dict["dataset"]["seed"]
    ignore_dev_set = exp_dict["ignore_dev_set"]
    batch_size = exp_dict["batch_size"]
    # if use_probs and temperature is None:
    #    raise Exception("use_probs==True, so a temperature must be set")

    k_shot = exp_dict["dataset"]["k_shot"]
    if not exp_dict["ignore_k_shot_warning"]:
        if k_shot != ae_exp_dict["dataset"]["k_shot"]:
            raise Exception(
                "k_shot for this experiment does not match the k_shot of GAN experiment"
            )

    dev_dataset = datasets.get_dataset(
        class_split="valid",
        which_set="supports",
        k_shot=k_shot,
        datadir=args.datadir,
        dataset=dataset_name,
        seed=dataset_seed,
        input_size=dataset_input_size,
        transform_kwargs=exp_dict["dataset"]["transform_kwargs"],
    )
    dev_loader, _ = datasets.get_loader(
        dev_dataset, batch_size=batch_size, num_workers=args.num_workers
    )

    valid_dataset = datasets.get_dataset(
        class_split="valid",
        which_set="valid",
        k_shot=k_shot,
        datadir=args.datadir,
        dataset=dataset_name,
        seed=dataset_seed,
        input_size=dataset_input_size,
    )
    valid_loader, _ = datasets.get_loader(
        valid_dataset, batch_size=batch_size, num_workers=args.num_workers
    )

    test_dataset = datasets.get_dataset(
        class_split="valid",
        which_set="test",
        k_shot=k_shot,
        datadir=args.datadir,
        dataset=dataset_name,
        seed=dataset_seed,
        input_size=dataset_input_size,
    )
    test_loader, _ = datasets.get_loader(
        test_dataset, batch_size=batch_size, num_workers=args.num_workers
    )

    logger.info("dev_dataset: {}".format(dev_dataset))
    logger.info("valid dataset: {}".format(valid_dataset))
    logger.info("test dataset: {}".format(test_dataset))

    # Load the pre-trained autoencoder.
    # ---------------------------------
    model = InfoGAN(
        input_size=dataset_input_size,
        exp_dict=ae_exp_dict,
        n_classes=dev_loader.dataset.n_classes,
        train=False,
        rank=0,
    )
    model_state_dict = torch.load(ae_exp_model_path)
    model.set_state_dict(model_state_dict, load_opt=False)

    # Also load pretrained classifier model
    # -------------------------------------
    optim_args = exp_dict["optim"]
    model_clf = models_clf.get_model(
        n_classes=dev_dataset.n_classes,
        freeze_all_except=exp_dict["freeze_all_except"],
        lr=optim_args["lr"],
        beta1=optim_args["beta1"],
        beta2=optim_args["beta2"],
        eps=optim_args["eps"],
        weight_decay=optim_args["weight_decay"],
        step_size=optim_args["step_size"],
        gamma=optim_args["gamma"],
    )
    clf_state_dict = torch.load(os.path.join(clf_exp_model_path))
    logger.info("Loading clf {} ...".format(clf_exp_dir))
    model_clf.set_state_dict(clf_state_dict, load_opt=False)

    # Load the classifier from its actual save dir, if needed
    # -------------------------------------------------------
    chk_dict = get_checkpoint(
        savedir,
        return_model_state_dict=True,
        map_location=lambda storage, loc: storage.cuda(0),
    )
    if len(chk_dict["model_state_dict"]):
        model.set_state_dict(chk_dict["model_state_dict"], strict=True)

    if dataset_seed != ae_exp_dict["dataset"]["seed"]:
        raise Exception(
            """ERROR: The dataset seed used to train the classifier {} is {}, but the seed used for the autoencoder {} is {} (which is also the seed we use for fine-tuning here). Since these seeds are not the same, it is possible that the classifier was pre-trained on classes in #the valid/dev set split, which means the fine-tuning results you get here will be optimistic / #inaccurate. Because of this, the script is terminating here. Please make sure you are using the #correct `clf_exp` and `ae_exp` arguments.
        """.format(
                clf_exp_dir, dataset_seed, ae_exp_dir, ae_exp_dict["dataset"]["seed"]
            )
        )

    # When haven saves exp_dict.json, it does not consider keys (default keys)
    # inserted after the experiment launches. So save the new exp_dict here.
    with open("{}/exp_dict.json".format(savedir), "w") as f:
        f.write(json.dumps(exp_dict))

    # Also save experiment metadata.
    # with open("{}/metadata.json".format(savedir), "w") as f:
    #    f.write(json.dumps({
    ##        'ae_epochs': len(score_list)
    #    }))

    n_samples_per_class = exp_dict["n_samples_per_class"]

    mode = exp_dict["mode"]

    epochs = exp_dict["epochs"]
    use_mixup = exp_dict["use_mixup"]
    transform_kwargs = exp_dict["finetune"]["transform_kwargs"]
    aug_transform = dev_dataset.get_transform_finetune(
        dataset_input_size, **transform_kwargs
    )
    N_fakeval = exp_dict['N_fakeval']
    logger.info("Transform for augmented samples: {}".format(aug_transform))

    logger.info("Generating fakeval dataset...")
    fakeval_dataset = dataset_utils.get_augmented_dataset_samples(
        dataset=dev_loader.dataset,
        model=model,
        transform=aug_transform,
        n_samples_per_class=N_fakeval,
        stdev=exp_dict["stdev"],
        mixup=False,
    )

    if mode == "baseline":

        actual_dataset = dev_loader.dataset

    elif mode == "sample":

        # Generate samples by sampling from the prior distribution
        # only.
        logger.info(
            "sample augment, n_c = {}, stdev = {}".format(
                n_samples_per_class, exp_dict["stdev"]
            )
        )
        augmented_dev_dataset = dataset_utils.get_augmented_dataset_samples(
            dataset=dev_loader.dataset,
            model=model,
            transform=aug_transform,
            n_samples_per_class=n_samples_per_class,
            stdev=exp_dict["stdev"],
            mixup=mode == "sample_mixup",
        )

        if ignore_dev_set:
            # If set, do not incorporate original supports
            actual_dataset = augmented_dev_dataset
        else:
            actual_dataset = ConcatDataset([augmented_dev_dataset, dev_loader.dataset])
    else:

        raise Exception("mode not recognised: {}".format(mode))

    if exp_dict["dataset"]["pad_length"] > 0:
        M = exp_dict["dataset"]["pad_length"]
        logger.info("pad_length is set, so pad dataset {} times".format(M))
        actual_dataset = DuplicateDatasetMTimes(actual_dataset, M=M)

    actual_loader = DataLoader(actual_dataset, batch_size=batch_size, shuffle=True)
    fakeval_loader = DataLoader(fakeval_dataset, batch_size=batch_size, shuffle=True)

    if mode == "baseline":
        with torch.no_grad():
            dev_loader_batch = iter(
                DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=True)
            ).next()
            logger.info("Saving real images to {}/real.png".format(savedir))
            viz_images(
                dev_loader_batch, outfile="{}/real.png".format(savedir), invert=False
            )
            viz_images(
                dev_loader_batch, outfile="{}/real-i.png".format(savedir), invert=True
            )
    else:
        with torch.no_grad():
            # Create a neat visualisation for the images.
            dev_loader_noshuffle = DataLoader(
                dev_dataset, batch_size=len(dev_dataset), shuffle=False
            )
            # TODO: aug_loader needs to be defined so the entire
            # dataset embeddings can be extracted.
            aug_loader = DataLoader(
                augmented_dev_dataset,
                batch_size=len(augmented_dev_dataset),
                shuffle=False,  # MUST BE FALSE
            )
            logger.info(
                "Length of augmented dataset: {}".format(len(augmented_dev_dataset))
            )
            aug_loader_batch = iter(aug_loader).next()
            dev_loader_batch = iter(dev_loader_noshuffle).next()
            logger.info("Saving vis-all to {}/all.png".format(savedir))
            viz_real_and_augmented_images(
                real_batch=dev_loader_batch,
                aug_batch=aug_loader_batch,
                class_to_metric=None,
                outfile="{}/all.png".format(savedir),
                invert=False,
            )
            viz_real_and_augmented_images(
                real_batch=dev_loader_batch,
                aug_batch=aug_loader_batch,
                outfile="{}/all-i.png".format(savedir),
                invert=True,
            )

    with open("{}/metadata.json".format(savedir), "w") as f:
        logger.info("metadata = {}".format(METADATA))
        f.write(json.dumps(METADATA))

    score_dir = "{}/predictions".format(savedir)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    chk_dict = {"score_list": []}
    if exp_dict["bn_moving_avgs"]:
        logger.info("bn moving avgs is set, so model_clf will be set to train mode...")
    for epoch in range(0, epochs):
        score_dict = {}
        score_dict["epoch"] = epoch

        if epoch == 0 and use_mixup:
            logger.info("train_on_loader invoked with input mixup")
        
        train_dev_dict = model_clf.train_on_loader(
            actual_loader,
            train_mode=exp_dict["bn_moving_avgs"],  # modify bn stats if flag set
            mixup=use_mixup,
            mixup_dist=exp_dict["mixup"]["dist"],
            mixup_alpha=exp_dict["mixup"]["alpha"],
            mixup_labels=exp_dict["mixup"]["mix_labels"],
        )
        train_dev_dict = {("train_" + k): v for k, v in train_dev_dict.items()}
        score_dict.update(train_dev_dict)

        validate_every = exp_dict["validate_every"]
        if epoch % validate_every == 0:

            valid_dict = model_clf.val_on_loader(valid_loader, desc="validating")
            valid_dict = {("valid_" + k): v for k, v in valid_dict.items()}
            score_dict.update(valid_dict)

            fakeval_dict = model_clf.val_on_loader(fakeval_loader, desc="fake validating")
            fakeval_dict = {("fakeval_" + k): v for k, v in fakeval_dict.items()}
            score_dict.update(fakeval_dict)

            test_dict = model_clf.val_on_loader(test_loader, desc="testing")
            test_dict = {("test_" + k): v for k, v in test_dict.items()}
            score_dict.update(test_dict)

        """
        if epoch % 10 == 0:
            pred_labels, actual_labels = model_clf.score_on_loader(
                valid_loader)
            np.savez("{}/{}".format(score_dir, epoch+1),
                     pred=pred_labels,
                     y=actual_labels)
        """

        chk_dict["score_list"] += [score_dict]

        # Save checkpoint
        if epoch % exp_dict["save_every"] == 0:
            hw.save_checkpoint(
                savedir,
                score_list=chk_dict["score_list"],
                model_state_dict=model.get_state_dict(),
                verbose=False,
            )

    print("Experiment completed")
