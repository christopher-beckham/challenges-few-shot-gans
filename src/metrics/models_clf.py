import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import tqdm

from .resnet18 import Resnet18

from functools import partial

try:
    from apex.optimizers import FusedAdam as AdamW

    Adam = partial(AdamW, adam_w_mode=True)
    # logger.info("Successfully imported fused Adam")
except:
    from torch.optim import AdamW as Adam

    # logger.warning("Unable to import fused AdamW, using default AdamW...")

from .. import setup_logger
logger = setup_logger.get_logger(__name__)

from torch.optim import lr_scheduler as lr_sched

from ..models import utils as ut

def get_model(n_classes, **kwargs):
    model = Resnet18(n_classes=n_classes)
    return Classifier(model=model, **kwargs)


# model definition
class Classifier:
    def __init__(
        self,
        model,
        freeze_all_except,
        lr=2e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        gamma=None,
        step_size=None,
        verbose=False,
    ):

        if not hasattr(model, "encode"):
            raise Exception("The `model` module must have an encode() method")

        self.model = model
        self.model.cuda()

        if freeze_all_except is not None:
            logger.info("Freeze all parameters in feature extractor...")
            # First freeze ALL parameters in
            # the feature extractor.
            for p in self.model._modules["f"].parameters():
                p.requires_grad = False
            f_modules = self.model._modules["f"]
            # -1 = flatten, -2 = adaptivepool
            # Now selectively unfreeze modules.
            for freeze_idx in freeze_all_except:
                idx1, idx2 = [int(x) for x in freeze_idx.split(":")]
                this_module = f_modules[idx1]
                logger.info("Unfreezing this module: {}".format(this_module[idx2]))
                for p in this_module[idx2].parameters():
                    p.requires_grad = True

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.opt = Adam(
            params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, eps=eps
        )

        self.sched = None
        if gamma is not None and step_size is not None:
            self.sched = lr_sched.StepLR(self.opt, step_size=step_size, gamma=gamma)
            logger.info("sched: {}".format(self.sched))
        n_p = ut.count_params(self.model, trainable_only=True)
        logger.info("ResNet: {} with {} trainable params".format(self.opt, n_p))

        self.loss = nn.CrossEntropyLoss()

    def set_state_dict(self, state_dict, load_opt=True):
        self.model.load_state_dict(state_dict["model"])
        if load_opt:
            self.opt.load_state_dict(state_dict["opt"])

    def get_state_dict(self):
        state_dict = {"model": self.model.state_dict(), "opt": self.opt.state_dict()}
        return state_dict

    @torch.no_grad()
    def extract_embeddings(self, loader, max_iters=None):
        self.model.eval()
        x_buf = []
        y_buf = []
        for b, batch in enumerate(tqdm.tqdm(loader, desc="Encoding")):
            if max_iters is not None and b > max_iters:
                break
            xi, yi = batch["images"][:, 0], batch["labels"][:, 0]
            xi = xi.cuda()
            yi = yi.cuda()
            features = self.model.encode(xi)
            x_buf.append(features)
            y_buf.append(yi)
        x_buf = torch.cat(x_buf, dim=0).cpu()
        y_buf = torch.cat(y_buf, dim=0).cpu()
        return x_buf, y_buf

    def cross_entropy_vector(self, y_pred, y_actual):
        return (-y_actual * torch.log(y_pred)).sum(dim=1)

    def _sample_uniform(self, batch_size, alpha):
        # NOTE: old implementation was just a single
        # scalar, so it was the same across all batches.
        return torch.zeros((batch_size, 1)).uniform_(0, alpha)

    def _sample_beta(self, batch_size, alpha):
        coefs = np.random.beta(alpha, alpha, size=(batch_size, 1))
        return torch.from_numpy(coefs).float()

    def _unsqueeze2(self, x):
        return x.unsqueeze(-1).unsqueeze(-1)

    def _train_on_instance_vanilla(self, x, y, y_probs):
        logits = self.model(x)
        y_pred = F.softmax(logits, dim=1)
        if y_probs is None:
            # Just use nn.CrossEntropyLoss()
            loss = nn.CrossEntropyLoss()(logits, y)
        else:
            loss = self.cross_entropy_vector(y_pred, y_probs).mean()
        return {
            'loss': loss,
            'pred': y_pred
        }

    def _train_on_instance_mixup(self, x, y_probs, 
                                 mixup_dist_fn, 
                                 mixup_alpha,
                                 mixup_labels):
        perm = torch.randperm(x.size(0))
        x2 = x[perm]
        y_probs2 = y_probs[perm]

        # alpha = np.random.uniform()
        alpha = mixup_dist_fn(x.size(0), mixup_alpha).cuda()

        x_mix = self._unsqueeze2(alpha) * x + (1 - self._unsqueeze2(alpha)) * x2
        # This is numerically unstable. Use the formulation instead:
        # lambda*loss(x_mix, y1) + (1-lambda)*loss(x_mix, y2)

        #if mixup_labels:
        #    y_mix = alpha * y_probs + (1 - alpha) * y_probs2
        #else:
        #    # otherwise, choose yi if alpha >= 0.5 else yi2
        #    #alpha_binary = (alpha >= 0.5).float()
        #    #y_mix = alpha_binary * y_probs + (1 - alpha_binary) * y_probs2
        #    raise NotImplementedError()
        
        #pred_mix = F.softmax(self.model(x_mix))
        #loss = self.cross_entropy_vector(pred_mix, y_mix).mean()
        logits = self.model(x_mix)
        cce = nn.CrossEntropyLoss(reduction='none')
        loss = alpha * ( cce(logits, y_probs.argmax(1)).view(-1, 1) ) + \
            (1-alpha) * ( cce(logits, y_probs2.argmax(1)).view(-1, 1) )        
        
        return {
            'loss': loss.mean(),
            'pred': F.softmax(logits)
        }

    # train the model
    def train_on_loader(
        self,
        loader,
        preprocessor=None,
        measure_grad=True,
        train_mode=True,
        mixup=False,
        mixup_dist=None,
        mixup_alpha=None,
        mixup_labels=True,
        savedir=None,
    ):

        if mixup:
            MIXUP_MODES = ["uniform", "beta"]
            if mixup_dist not in MIXUP_MODES:
                raise Exception("mixup mode must be one of: {}".format(MIXUP_MODES))
            if type(mixup_alpha) != float:
                raise Exception("mixup_alpha must be a float")
            if mixup_dist == "uniform":
                mixup_dist_fn = self._sample_uniform
            else:
                mixup_dist_fn = self._sample_beta

        if train_mode:
            # Compute BN estimates via minibatch but
            # also update the moving averages.
            self.model.train()
        else:
            # This should be eval for fine-tuning, since
            # we do not want to use the moving avg BN
            # estimates.
            self.model.eval()

        if preprocessor is None:
            # preprocessor is used if we would like to, for instance,
            # train on the reconstructions of the images instead, e.g.
            # we could end up doing self.model(ae.reconstruct(x))
            preprocessor = lambda x: x

        losses = []
        accs = []

        max_grad_seen = torch.FloatTensor([-1.]).cuda()
        for b, batch in enumerate(tqdm.tqdm(loader, desc="Training")):

            self.opt.zero_grad()

            # TODO: verify experiments in v1, it looks like
            # the data aug experiments used 'images' instead
            # of 'images_aug'.
            xi = batch["images_aug"][:, 0]
            yi = batch["probs"][:, 0, :].argmax(dim=1)
            yi_probs = batch["probs"][:, 0, :].cuda()

            if b == 0 and savedir is not None:
                save_image(xi * 0.5 + 0.5, "{}/xi.png".format(savedir))

            xi = xi.cuda()
            yi = yi.cuda()

            xi = preprocessor(xi)
            n_classes = yi_probs.size(1)

            if mixup:
                if b == 0:
                    logger.info("mixup mode set, with label mixing={}".format(mixup_labels))
                metrics = self._train_on_instance_mixup(xi, yi_probs,
                                                        mixup_dist_fn=mixup_dist_fn,
                                                        mixup_alpha=mixup_alpha,
                                                        mixup_labels=mixup_labels)
            else:
                # Check if yi_probs is one hot, if so, don't pass into yi_probs
                # (so train_on_instance_vanilla can use the numerically stable
                # softmax loss).
                if torch.all( torch.eye(n_classes)[yi].cuda() == yi_probs ):
                    if b == 0:
                        logger.info("detected that argmax(y_probs, 1) == y, so using " + \
                                    "nn.CrossEntropyLoss()")
                    metrics = self._train_on_instance_vanilla(xi, yi, None)
                else:
                    if b == 0:
                        logger.info("detected that argmax(y_probs, 1) != y, so using " + \
                            "self.cross_entropy_vector()")
                    metrics = self._train_on_instance_vanilla(xi, yi, yi_probs)

            loss = metrics['loss']            
            loss.backward()

            if measure_grad:
                with torch.no_grad():
                    max_grad = max([ torch.max(p.grad**2) for p in self.opt.param_groups[0]['params'] ])
                    if max_grad_seen < max_grad:
                        max_grad_seen.data.mul_(0).add_(max_grad)
                        #logger.info("max_grad_seen updated to {}".format(max_grad))
            
            self.opt.step()

            yhat = metrics['pred']
            losses.append(metrics['loss'].item())

            with torch.no_grad():
                # NOTE: if the label isn't one-hot-encoded (like when
                # mixed labels are using during training) then this
                # won't make much sense. Of course, for validation
                # if you're using one-hot labels it's fine.
                acc = (yhat.argmax(dim=1) == yi).float().mean()
                accs.append(acc.item())

        if self.sched is not None:
            self.sched.step()

        return {"loss": np.mean(losses), 
                "acc": np.mean(accs),
                "max_grad_seen": max_grad_seen.item() }

    # evaluate the model
    @torch.no_grad()
    def val_on_loader(self, loader, preprocessor=None, desc="Validating"):
        self.model.eval()

        if preprocessor is None:
            # preprocessor is used if we would like to, for instance,
            # train on the reconstructions of the images instead, e.g.
            # we could end up doing self.model(ae.reconstruct(x))
            preprocessor = lambda x: x

        accs = []
        for b, batch in enumerate(tqdm.tqdm(loader, desc=desc)):

            xi, yi = batch["images"][:, 0], batch["labels"][:, 0]

            xi = xi.cuda()
            yi = yi.cuda()

            xi = preprocessor(xi)

            yhat = self.model(xi)
            acc = (yhat.argmax(dim=1) == yi).float().mean()

            accs.append(acc.item())

        return {"acc": np.mean(accs)}

    @torch.no_grad()
    def score_on_loader(self, loader, max_iters=None):
        # preds = []
        probs = []
        labels = []
        for b, batch in enumerate(tqdm.tqdm(loader, desc="Scoring")):

            xi, yi = batch["images"][:, 0], batch["labels"][:, 0]

            xi = xi.cuda()
            yi = yi.cuda()

            yhat = self.model(xi)
            # pred = yhat.argmax(dim=1)

            # preds.append(pred)
            probs.append(yhat)
            labels.append(yi)

            if max_iters is not None and b > max_iters:
                break

        probs = torch.cat(probs, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

        return probs, labels
