import os
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from collections import OrderedDict, Counter
from functools import partial

from . import utils as ut
from .utils import (Argument,
                    validate_and_insert_defaults)
from . import utils_viz
from .sg2_ada_augment import AugmentPipe
from .networks import main, cosgrove
from ..setup_logger import get_logger

logger = get_logger(__name__)

if ut.str2bool(os.environ.get("ENABLE_APEX", "")):
    try:
        from apex.optimizers import FusedAdam as AdamW

        Adam = partial(AdamW, adam_w_mode=True)
        logger.info("Successfully imported fused Adam")
    except:
        from torch.optim import AdamW as Adam

        logger.warning("Unable to import fused AdamW, using default AdamW...")
else:
    from torch.optim import AdamW as Adam

ENABLE_FLOAT16 = False

class WrapperModule(nn.Module):
    def __init__(self, gen, disc, probe, container):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.probe = probe
        self.container = container


class DistributedProbeLosses(DDP):
    @autocast(enabled=ENABLE_FLOAT16)
    def forward(self, x_dev, y_dev, epoch=None):

        gen = self.module.gen
        probe = self.module.probe
        # This is a reference to the AEMixup class,
        # which we will need here.
        ctr = self.module.container
        cce = nn.CrossEntropyLoss()

        r_x = gen.encode(x_dev[:, 0])
        pred = probe(r_x.detach())
        probe_loss = cce(pred, y_dev)

        with torch.no_grad():
            probe_acc = (pred.argmax(dim=1) == y_dev).float().mean()

        p_loss_dict = OrderedDict({})
        p_loss_dict["probe_loss"] = (1.0, probe_loss)

        return p_loss_dict, {}, {"probe_acc": probe_acc}


def simsiam_loss(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


class DistributedGeneratorLosses(DDP):
    """Distributed data parallel requires
    that all the logic goes inside forward(),
    so we need to do all the loss computation
    here.

    Args:
        DDP ([type]): [description]
    """

    @autocast(enabled=ENABLE_FLOAT16)
    def forward(self, x_real, y_real, semi_sup=False, epoch=None):
        """
        There are three losses:
        - (1) GAN loss, make sure fakes are indistinguishable
            from real. This loss is conditioned on the label.
        - (2) InfoGAN losses, max mutual info between G(r,z) and
            (r, z).
        - (3) SimSIAM loss, make sure that r_x embeddings of same-class
            examples are closer to each other than ones from different
            classes. This loss was added to help facilitate the rejection
            sampling portion of fine-tuning, where we would like to
            filter out generated examples that are not class-consistent.

        Args:
            tgt_real ([type]): [description]
            tgt_labels ([type]): [description]
            epoch ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        gen = self.module.gen
        disc = self.module.disc
        # This is a reference to the AEMixup class,
        # which we will need here.
        ctr = self.module.container
        pipe = ctr.augment_pipe
        bce = nn.BCEWithLogitsLoss()

        ones = torch.ones((x_real.size(0), 1)).float().to(ctr.rank)
        # zeros = torch.zeros((x_real.size(0), 1)).float().to(ctr.rank)

        g_loss_dict = OrderedDict({})

        this_z = ctr.sample_z(x_real.size(0))
        # r_x.detach() because the generator (decoder)
        # training is separate from SimSiam.
        x_fake = gen.decode(y_real, this_z)

        # x_fake = G(r1, z)
        # we want D(x_fake|r{1,2}) to be 'real'
        d_out, d_out_rz = disc(pipe(x_fake), y_real)
        g_d_loss = bce(d_out, ones)
        mi_loss_rz = torch.mean((d_out_rz[1] - this_z) ** 2)

        if ctr.alpha > 0:
            d_out_uncond, _ = disc(pipe(x_fake), y=None)
            g_d_uncond_loss = bce(d_out_uncond, ones)
            g_loss_dict["g_d_uncond_loss"] = (ctr.alpha, g_d_uncond_loss)

        g_loss_dict["g_d_loss"] = (1.0, g_d_loss)
        g_loss_dict["g_mi_loss"] = (ctr.gamma, mi_loss_rz)
        metrics = {}

        outputs = {
            "x_fake": x_fake,
            #'x_r': y_embed,
            "x_z": this_z,
            #'r_acc': r_acc
        }

        return g_loss_dict, outputs, metrics


class DistributedDiscriminatorLosses(DDP):
    """Distributed data parallel requires
    that all the logic goes inside forward(),
    so we need to do all the loss computation
    here.

    Args:
        DDP ([type]): [description]
    """

    @autocast(enabled=ENABLE_FLOAT16)
    def forward(
        self,
        x_real,
        y_real,
        x_unsup,
        x_fake,
        # x_fake_r,
        x_fake_z,
        epoch=None,
    ):

        disc = self.module.disc
        # This is a reference to the AEMixup class,
        # which we will need here.
        ctr = self.module.container
        pipe = ctr.augment_pipe

        d_loss_dict = OrderedDict({})

        bce = nn.BCEWithLogitsLoss()
        zeros = torch.zeros((x_real.size(0), 1)).float().to(ctr.rank)
        ones = torch.ones((x_real.size(0), 1)).float().to(ctr.rank)

        # r_x1_aug = gen.encode(x_real[:,0]).detach()
        x1 = x_real[:, 0]

        d_out_real, _ = disc(pipe(x1), y_real)
        d_out_fake, d_out_fake_rz = disc(pipe(x_fake), y_real)

        mi_loss_rz = torch.mean((d_out_fake_rz[1] - x_fake_z) ** 2)
        d_loss_real = bce(d_out_real, ones)
        d_loss_fake = bce(d_out_fake, zeros)
        d_loss = (d_loss_real + d_loss_fake) / 2.0

        d_loss_dict["d_loss"] = (1.0, d_loss)
        d_loss_dict["d_mi_loss"] = (ctr.gamma, mi_loss_rz)

        # zeros_uc = torch.zeros((xu.size(0), 1)).float().to(ctr.rank)

        # xu = x_unsup[:, 0]
        # xu = x1
        if x_unsup is not None:
            # If semi-sup mode is set, then this will be an example
            # in the valid set.
            xu = x_unsup[:,0]
        else:
            xu = x1

        if ctr.alpha > 0:

            ones_uc = torch.ones((xu.size(0), 1)).float().to(ctr.rank)
            d_out_uncond_real, _ = disc.forward(pipe(xu), y=None)
            d_out_uncond_fake, _ = disc.forward(pipe(x_fake), y=None)
            d_loss_uncond_real = bce(d_out_uncond_real, ones_uc)
            d_loss_uncond_fake = bce(d_out_uncond_fake, zeros)
            d_loss_uncond = (d_loss_uncond_real + d_loss_uncond_fake) / 2.0
            d_loss_dict["d_uncond_loss"] = (ctr.alpha, d_loss_uncond)

        metrics = {
            "d_loss_dict": d_loss_dict,
        }
        return metrics

NONETYPE = type(None)

class InfoGAN:

    DEFAULTS = {
        "use_ema": Argument('use_ema', False, [bool]), # not used anymore
        "name": Argument('name', None, [str, NONETYPE]), # not used anymore
        # Generator args
        "ngf_decoder": Argument('ngf_decoder', 64, [int]), # channel multiplier for G
        "blocks_per_res": Argument('blocks_per_res', [1, 1, 1, 1], [list]),
        "z_dim": Argument('z_dim', 64, [int]),
        "n_downsampling": Argument('n_downsampling', 4, [int]),
        # Discriminator stuff.
        "alpha": Argument('alpha', 0.0, [float]),
        "ndf": Argument('ndf', 16, [int]), # channel multiplier for D
        "augment_p": Argument('augment_p', 0.0, [float]),
        "d_finetune_mode": Argument("d_finetune_mode", "embed", [str]),
        "g_finetune_mode": Argument("g_finetune_mode", "linear", [str]),
        # Coefficients.
        "pretrain": Argument("pretrain", False, [bool]), # not used anymore
        "gamma": Argument('gamma', 1.0, [float]), # infogan loss coefficient
        # args that are no longer used, added as dummy args to stop
        # script from crashing:
        "beta": Argument('beta', 0.0, [float]), # not used anymore
        "use_tanh": Argument("use_tanh", False, [bool]), # not used anymore
    }

    def _validate_args(self, dd):
        assert dd["d_finetune_mode"] in ["embed", "linear", "all"]
        assert dd["g_finetune_mode"] in ["embed", "linear", "all"]

    def __init__(
        self,
        input_size,
        n_classes,
        exp_dict,
        rank,
        train=True,
        finetune=False,
        verbose=True,
    ):
        super().__init__()
        self.exp_dict = exp_dict

        self.rank = rank

        # Validate arguments.
        logger.info("Validating and inserting defaults...")
        validate_and_insert_defaults(exp_dict["model"], self.DEFAULTS)
        self._validate_args(exp_dict["model"])

        self.use_ema = exp_dict["model"]["use_ema"]
        self.EPS = 1e-6

        self.scaler = GradScaler(enabled=ENABLE_FLOAT16)

        # TODO
        self.gen = main.GeneratorResNet(
            input_size=input_size,
            input_nc=3,
            ngf_decoder=exp_dict["model"]["ngf_decoder"],
            blocks_per_res=exp_dict["model"]["blocks_per_res"],
            z_dim=exp_dict["model"]["z_dim"],
            n_downsampling=exp_dict["model"]["n_downsampling"],
            n_classes=n_classes,
            # proj_dim=exp_dict['model']['proj_dim']
        )

        self.disc = cosgrove.DiscriminatorImage(
            input_nc=3,
            nf=exp_dict["model"]["ndf"],
            z_dim=exp_dict["model"]["z_dim"],
            n_classes=n_classes,
        )

        self.probe = nn.Linear(512, n_classes)

        self.gen.to(self.rank)
        self.disc.to(self.rank)
        self.probe.to(self.rank)

        if train:
            self.g_model = DistributedGeneratorLosses(
                WrapperModule(self.gen, self.disc, self.probe, self),
                device_ids=[self.rank],
            )
            self.d_model = DistributedDiscriminatorLosses(
                WrapperModule(self.gen, self.disc, self.probe, self),
                device_ids=[self.rank],
            )
            self.p_model = DistributedProbeLosses(
                WrapperModule(self.gen, self.disc, self.probe, self),
                device_ids=[self.rank],
            )
        else:
            # If we're in inference mode (i.e. fine-tuning)
            # then all the multi-GPU overhead stuff like
            # wrapped DDP objects (like self.x_model) are
            # disabled. This means run_on_batch will not
            # work, but other methods will.
            self.g_model = None

        opt_dict = exp_dict.get("optim", dict())
        this_lr = opt_dict.get("lr", 2e-4)
        this_beta1 = opt_dict.get("beta1", 0.9)
        this_beta2 = opt_dict.get("beta2", 0.999)
        this_wd = opt_dict.get("weight_decay", 0.0)
        this_eps = opt_dict.get("eps", 1e-8)
        self.n_gen = opt_dict.get("n_gen", 1)

        if finetune:
            d_finetune_mode = exp_dict["model"]["d_finetune_mode"]
            logger.info("Finetuning D with mode: {}".format(d_finetune_mode))
            if d_finetune_mode == "all":
                d_params = self.disc.parameters()
            elif d_finetune_mode == "linear":
                d_params = self.disc.finetune_parameters(embed_only=False)
            elif d_finetune_mode == "embed":
                d_params = self.disc.finetune_parameters(embed_only=True)
            g_finetune_mode = exp_dict["model"]["g_finetune_mode"]
            logger.info("Finetuning G with mode: {}".format(g_finetune_mode))
            if g_finetune_mode == "linear":
                g_params = self.gen.finetune_parameters(embed_only=False)
            elif g_finetune_mode == "embed":
                g_params = self.gen.finetune_parameters(embed_only=True)
            elif g_finetune_mode == "all":
                g_params = self.gen.parameters()
        else:
            g_params = self.gen.parameters()
            d_params = self.disc.parameters()
        p_params = self.probe.parameters()

        if verbose and self.rank == 0:
            if 'VERBOSE' in os.environ and os.environ['VERBOSE']=='1':
                logger.info("gen: {}".format(self.gen))
                logger.info("disc: {}".format(self.disc))
            logger.info("gen params: {}".format(ut.count_params(self.gen)))
            logger.info("disc params: {}".format(ut.count_params(self.disc)))

        self.opt_g = Adam(
            g_params,
            lr=this_lr,
            betas=(this_beta1, this_beta2),
            weight_decay=this_wd,
            eps=this_eps,
        )
        self.opt_d = Adam(
            d_params,
            lr=this_lr,
            betas=(this_beta1, this_beta2),
            weight_decay=this_wd,
            eps=this_eps,
        )
        self.opt_p = Adam(
            p_params,
            lr=this_lr,
            betas=(this_beta1, this_beta2),
            weight_decay=this_wd,
            eps=this_eps,
        )

        # self.alpha = exp_dict['model']['alpha']
        self.pretrain = exp_dict["model"]["pretrain"]
        self.beta = exp_dict["model"]["beta"]
        self.gamma = exp_dict["model"]["gamma"]
        self.alpha = exp_dict["model"]["alpha"]

        # BUG: if verbosity='brief' for stylegan2-ada-pytorch/torch_utils/custom_ops.py
        # then we get "Failed!" as an extra message, even if the augmentation pipeline
        # works. If verbosity==full then we get the following 'error':
        #   No modifications detected for re-loaded extension module upfirdn2d_plugin,
        #       skipping build step...
        #   Loading extension module upfirdn2d_plugin...
        # augment_pipe still works, but maybe this is an innocent compile error.
        augment_p = exp_dict['model']['augment_p']
        if augment_p > 0:
            self.augment_pipe = AugmentPipe(xflip=1, rotate90=1, xint=1, rotate=1).to(
                self.rank
            )
            self.augment_pipe.p *= 0.0
            self.augment_pipe.p += augment_p
            logger.info("augment_pipe.p = {}".format(self.augment_pipe.p))
        else:
            self.augment_pipe = nn.Identity()

        self.iteration = 0

    def _eval_loss_dict(self, loss_dict):
        loss = 0.0
        loss_str = []
        for key, val in loss_dict.items():
            if len(val) != 2:
                raise Exception("val must be a tuple of (coef, loss)")
            if val[0] != 0:
                # Only add the loss if the coef is != 0
                loss += val[0] * val[1]
            loss_str.append("%f * %s" % (val[0], key))
        return loss, (" + ".join(loss_str))

    def update_ema(self, ema_rate=0.999):
        for p1, p2 in zip(self.gen.parameters(), self.gen_ema.parameters()):
            p2.data.mul_(ema_rate)
            p2.data.add_(p1.data * (1 - ema_rate))

    def sample_z(self, bs, stdev=1.0):
        z = torch.zeros((bs, self.gen.z_dim)).normal_(0, stdev).to(self.rank)
        return z

    @torch.no_grad()
    def sample(self, y, stdev=1.0, denorm=True):
        # r_x = self.gen.encode(y)
        z = self.sample_z(y.size(0), stdev=stdev)
        xfake = self.gen.decode(y, z)
        if denorm:
            return self.denorm(xfake)
        else:
            return xfake

    @torch.no_grad()
    def sample_mixup(self, y1, y2, alpha, stdev=1.0, denorm=True):
        assert alpha.ndim == 2
        # r_x = self.gen.encode(y)
        z = self.sample_z(y1.size(0), stdev=stdev)
        xfake = self.gen.decode_mixup(y1, y2, alpha, z)
        if denorm:
            return self.denorm(xfake)
        else:
            return xfake

    @torch.no_grad()
    def generate_on_batch(self, images_, labels):
        """ """
        self.eval()

        gen = self.gen

        if len(images_.shape) <= 4:
            raise Exception("`images_` is expected to be of shape (bs, k, f, h, w)")

        # same class
        # (bs, k, f, h, w)
        images = images_.to(self.rank)
        labels = labels.to(self.rank)

        real = images[:, 0]

        # BUG: doesn't work on real[0:1], cos of this:
        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/47

        actn1 = gen.encode(real)
        x_mix, _ = gen.decoder(actn1, labels, get_latents=True)
        x_mix = gen.sample(x_mix)

        return x_mix.cpu()

    def train(self):
        self.gen.train()
        self.disc.train()
        self.probe.train()

    def eval(self):
        self.gen.eval()
        self.disc.eval()
        self.probe.eval()

    def _run_on_batch_probe(self, batch, train=True, savedir=None, **kwargs):
        if train:
            self.train()
        else:
            self.eval()

        if train:
            self.opt_p.zero_grad()

        p_model = self.p_model

        # (bs, 2, 3, h, w)
        x_real = batch["images_aug"].to(self.rank)
        y_real = batch["labels"][:, 0].to(self.rank)

        p_loss_dict, _, p_metrics = p_model(x_real, y_real, epoch=kwargs["epoch"] + 1)

        if train:
            p_total_loss, p_total_loss_str = self._eval_loss_dict(p_loss_dict)
            p_total_loss.backward()
            self.opt_p.step()

        with torch.no_grad():
            # TODO
            metrics = {k: v[1].detach() for k, v in p_loss_dict.items()}
            metrics.update({k: v.detach() for k, v in p_metrics.items()})

        return metrics

    def _run_on_batch(
        self, batch, batch_unsup, train=True, savedir=None, classes=None, **kwargs
    ):

        if train:
            self.train()
        else:
            self.eval()

        if train:
            self.opt_g.zero_grad()
            self.opt_d.zero_grad()

        g_model = self.g_model
        d_model = self.d_model

        metrics = {}

        #####################
        # Train the generator
        #####################

        # (bs, 3, 3, h, w)
        x_real = batch["images_aug"].to(self.rank)
        # x_real_nonaug = batch['images'].to(self.rank)
        y_real = batch["labels"][:, 0].to(self.rank)
        if batch_unsup is None:
            x_unsup = None
        else:
            x_unsup = batch_unsup["images_aug"].to(self.rank)
            if self.iteration == 1:
                logger.info("x_unsup is defined")

        # tmp0 = self.generate(y_real)

        if self.iteration == 1:
            save_image(x_real[:, 0] * 0.5 + 0.5, "{}/x0.png".format(savedir))
            save_image(x_real[:, 1] * 0.5 + 0.5, "{}/x1.png".format(savedir))
            save_image(x_real[:, 2] * 0.5 + 0.5, "{}/x2.png".format(savedir))
            if batch_unsup is not None:
                save_image(x_unsup[:, 0] * 0.5 + 0.5, "{}/xu.png".format(savedir))
            xpipe = self.augment_pipe(x_real[:, 0])
            save_image(xpipe * 0.5 + 0.5, "{}/xpipe.png".format(savedir))

        g_loss_dict, g_output, g_metrics = g_model(
            x_real,
            y_real,
            semi_sup=True if x_unsup is not None else False,
            epoch=kwargs["epoch"] + 1,
        )

        if train:
            g_total_loss, g_total_loss_str = self._eval_loss_dict(g_loss_dict)

            if self.iteration == 0:
                logger.info(
                    "{}: G is optimising this total loss: {}".format(
                        self.rank, g_total_loss_str
                    )
                )
                logger.info("x_real.shape = {}".format(x_real.shape))
                if not self.pretrain:
                    logger.info("x_fake.shape = {}".format(g_output["x_fake"].shape))

            if self.iteration % self.n_gen == 0:
                g_total_loss.backward()
                self.opt_g.step()

        #########################
        # Train the discriminator
        #########################

        self.opt_d.zero_grad()

        x_fake = g_output["x_fake"]
        # x_r = g_output['x_r']
        x_z = g_output["x_z"]

        d_output = d_model(
            x_real,
            y_real,
            x_unsup,
            x_fake.detach(),
            # x_r.detach(),
            x_z.detach(),
            epoch=kwargs["epoch"] + 1,
        )

        d_loss_dict = d_output["d_loss_dict"]

        if train:
            d_total_loss, d_total_loss_str = self._eval_loss_dict(d_loss_dict)
            if self.iteration == 0:
                logger.info(
                    "{}: D is optimising this total loss: {}".format(
                        self.rank, d_total_loss_str
                    )
                )

            d_total_loss.backward()
            self.opt_d.step()

        self.iteration += 1

        # TODO: make this morboe efficient
        with torch.no_grad():
            metrics = {k: v[1].detach() for k, v in g_loss_dict.items()}
            if not self.pretrain:
                metrics.update({k: v[1].detach() for k, v in d_loss_dict.items()})

            metrics.update({k: v.detach() for k, v in g_metrics.items()})
            # metrics['r_loss_neg'] = r_loss_neg.detach()
            # metrics['r_acc'] = r_acc.detach()

        if self.iteration % 200 == 0:
            for k, v in metrics.items():
                logger.info("{} = {}".format(k, v))

        return metrics

    def denorm(self, x):
        return x * 0.5 + 0.5

    def vis_on_loader(
        self, loader, savedir, split, n_batches=1, aux_loader=None, **kwargs
    ):
        for i, batch in enumerate(loader):
            self.vis_on_batch(
                batch, savedir=savedir, split=split, dataset=loader.dataset
            )
            break

    def _run_on_loader(
        self,
        run_on_batch_fn,
        loader,
        train,
        unsup_loader=None,
        savedir=None,
        pbar=True,
        **kwargs
    ):
        # Reset the iteration number
        # train_list = []
        if savedir is not None:
            logger.info("savedir: {}".format(savedir))
        buf = {}
        desc_str = "train" if train else "validate"
        if unsup_loader is not None:
            unsup_loader_iter = iter(unsup_loader)
        for b, batch in enumerate(tqdm.tqdm(loader, desc=desc_str, disable=not pbar)):
            batch_unsup = None
            if unsup_loader is not None:
                # Support a semi-supervised batch if needed.
                try:
                    batch_unsup = unsup_loader_iter.next()
                except StopIteration:
                    unsup_loader_iter = iter(unsup_loader)
                    batch_unsup = unsup_loader_iter.next()
                del batch_unsup["labels"]
            train_dict = run_on_batch_fn(
                batch, batch_unsup, train=train, savedir=savedir, iter=b, **kwargs
            )
            # train_list += [train_dict]
            for key in train_dict:
                if key not in buf:
                    buf[key] = []
                buf[key].append(train_dict[key])

        # if savedir is not None:
        #    # If we have passed a save dir, then also invoke
        #    # vis_on_batch to visualise what the model is
        #    # doing.
        #    self.vis_on_batch(batch, savedir=savedir, split='train')

        pd_metrics = {}
        for key, val in buf.items():
            if type(val[0]) == torch.Tensor:
                pd_metrics[key] = torch.stack(val).detach().cpu().numpy()
            else:
                pd_metrics[key] = np.asarray(val)

        all_metrics = {k + "_mean": v.mean() for k, v in pd_metrics.items()}
        all_metrics.update({k + "_min": v.min() for k, v in pd_metrics.items()})
        all_metrics.update({k + "_max": v.max() for k, v in pd_metrics.items()})

        return all_metrics

    def train_on_loader(self, loader, unsup_loader=None, savedir=None, **kwargs):
        return self._run_on_loader(
            self._run_on_batch,
            loader,
            train=True,
            unsup_loader=unsup_loader,
            savedir=savedir,
            **kwargs
        )

    def eval_on_loader(self, loader, savedir=None, **kwargs):
        return self._run_on_loader(
            self._run_on_batch, loader, train=False, savedir=savedir, **kwargs
        )

    def train_probe_on_loader(self, loader, savedir=None, **kwargs):
        return self._run_on_loader(
            self._run_on_batch_probe, loader, train=True, savedir=savedir, **kwargs
        )

    def eval_probe_on_loader(self, loader, savedir=None, **kwargs):
        return self._run_on_loader(
            self._run_on_batch_probe, loader, train=False, savedir=savedir, **kwargs
        )

    @torch.no_grad()
    def extract_embeddings(self, loader, use_pbar=True, max_iters=None):
        buf = []
        all_labels = []
        for b, batch in enumerate(tqdm.tqdm(loader, disable=not use_pbar)):
            if max_iters is not None and b > max_iters:
                break
            imgs = batch["images"][:, 0].to(self.rank)
            labels = batch["labels"][:, 0]
            embeds = self.gen.encode(imgs).cpu()
            buf.append(embeds)
            all_labels.append(labels)
        buf = torch.cat(buf, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return buf, all_labels

    @torch.no_grad()
    def sample_from_loader(self, loader, N=None, label=None, verbose=True):
        """Sample images using data loader. This works by
        sampling minibatches, extracting labels, and using
        those labels to generate via the generator's p(x|y).

        Returns a `torch.Tensor` of generated images. These
        images are already de-normalised, since `sample` performs
        this operation.

        Args:
            loader (torch.utils.data.DataLoader): loader
            N (int): max number of images to generate
        """
        buf = []
        n_generated = 0
        if verbose:
            # record labels
            buf_labels = []
        for batch in loader:
            labels = batch["labels"][:, 0].to(self.rank)
            if label is not None:
                # if a label is specified, use that instead
                labels = labels * 0 + label
            generated = self.sample(labels, denorm=False).cpu()
            buf.append(generated)
            if verbose:
                buf_labels.append(labels)
            n_generated += generated.size(0)
            if N is not None:
                if n_generated >= N:
                    break
        buf = torch.cat(buf, dim=0)[0:N]
        if verbose:
            buf_labels = torch.cat(buf_labels, dim=0)
            logger.info("class stats: {}".format(Counter(buf_labels.cpu().numpy())))
        return buf

    def get_state_dict(self):
        state_dict = {
            "gen": self.gen.state_dict(),
            "disc": self.disc.state_dict(),
            "probe": self.probe.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }

        return state_dict

    def _load_state_dict_with_mismatch(self, current_model_dict, chkpt_model_dict):
        # https://github.com/pytorch/pytorch/issues/40859
        # strict won't let you load in a state dict with
        # mismatch param shapes, so we do this hack here.
        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
            for k, v in zip(current_model_dict.keys(), chkpt_model_dict.values())
        }
        return new_state_dict

    def set_state_dict(self, state_dict, load_opt=True, strict=True):
        if strict:
            self.gen.load_state_dict(state_dict["gen"], strict=strict)
        else:
            self.gen.load_state_dict(
                self._load_state_dict_with_mismatch(
                    current_model_dict=self.gen.state_dict(),
                    chkpt_model_dict=state_dict["gen"],
                )
            )
        self.disc.load_state_dict(state_dict["disc"], strict=strict)
        self.probe.load_state_dict(state_dict["probe"], strict=strict)

        if load_opt:
            self.opt_g.load_state_dict(state_dict["opt_g"])
            self.opt_d.load_state_dict(state_dict["opt_d"])
        # self.opt_p.load_state_dict(state_dict['opt_p'])

    @torch.no_grad()
    def vis_on_batch(self, batch, savedir, split, **kwargs):

        self.eval()

        images = batch["images"].to(self.rank)
        labels = batch["labels"][:, 0].to(self.rank)

        imgs1 = images[:, 0]
        perm = torch.randperm(imgs1.size(0))
        imgs2 = imgs1[perm]
        labels1 = labels
        labels2 = labels[perm]

        dataset = kwargs["dataset"]

        if not os.path.exists(os.path.join(savedir, "images")):
            os.makedirs(os.path.join(savedir, "images"))

        if split != "train":
            pass
            # print("split != `train`, so setting labels1 and labels2 to `None`...")
            # print("This means that the AE will try to infer these labels")

        # generate images
        utils_viz.sample(
            self,
            dataset,
            bs=labels1.size(0),
            savedir=os.path.join(savedir, "images", "{}_generated.png".format(split)),
        )

        utils_viz.viz_mixup_crossover(
            self,
            imgs1,
            imgs2,
            labels1,
            labels2,
            savedir=os.path.join(
                savedir, "images", "{}_interp_xover.png".format(split)
            ),
        )

        utils_viz.viz_mixup_class_interpolation(
            self,
            labels1,
            labels2,
            imgs1.size(-1),
            savedir=os.path.join(
                savedir, "images", "{}_class_interp.png".format(split)
            ),
        )

        """
        utils_viz.viz_reencoding(
            self,
            imgs1,
            labels1,
            savedir=os.path.join(
                savedir, 
                'images', 
                "{}_reenc.png".format(split)
            )
        )
        """
