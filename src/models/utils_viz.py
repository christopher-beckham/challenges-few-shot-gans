import torch
import torchvision
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import tempfile
import imageio

def _annotate_image(image,
                    im_size,
                    pad_size,
                    labels):
    
    pil_img = Image.fromarray(
        (image.permute(1,2,0)*255.).numpy().astype(np.uint8) 
    )
    
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    for j in range(len(labels)):
        if labels[j] is None:
            continue
        text = "{}".format(labels[j])
        wt, ht = font.getsize(text)
        yy = (im_size+pad_size)*j
        draw.rectangle( (0, yy, 0+wt, yy+ht), fill='white' )
        draw.text(
            (0, yy),
            text,
            (0,0,255)
        )
    return pil_img


@torch.no_grad()
def viz_reconstruction(model, imgs, savedir):
    
    img = imgs[:,0]
    img2 = imgs[:,1]
    
    enc = model.gen.encode(img)
    recon_imgs = model.gen.sample(
        model.gen.decode(enc)[0]
    )    
    
    
    im_size = img.size(-1)
    pad_size = 2

    layer_stats = model.gen.forward_get_latents(img)
    latents = [stat['z'] for stat in layer_stats]
    
    sampled = []
    for k in range(len(latents)):
        sampled_img = model.gen.sample(
            model.gen.forward_samples_set_latents(
                n_batch=img.size(0),
                latents=latents[0:k]
            )
        )         
        sampled.append(sampled_img)
    
    pad = recon_imgs*0.
    
    all_imgs = [img, img2, recon_imgs, pad] + sampled
    
    images = torchvision.utils.make_grid(
        model.denorm(torch.cat(all_imgs, dim=0)),
        nrow=imgs.shape[0]
    )
    
    save_image(model.denorm(torch.cat(all_imgs, dim=0)),
                savedir + ".tmp.png")
    
    
    pil_img = _annotate_image(
        images,
        im_size=im_size,
        pad_size=pad_size,
        labels=["x1", "x2", "rec(x1)"] + \
            ["posterior 0:k, prior k::"] + np.arange(len(latents)).tolist()
    )
    pil_img.save(savedir)
    
    return images


def viz_mixup_crossover(model,
                        imgs1,
                        imgs2,
                        labels1,
                        labels2,
                        savedir,
                        **kwargs):
    
    #model.use_ema = False

    # TODO: be able to set the seed

    model.eval()
    
    # Use the same reference image for each column.
    imgs1 = imgs1[0:1].repeat(imgs2.size(0), 1, 1, 1)
    if labels1 is not None:
        labels1 = labels1[0:1].repeat(imgs2.size(0))

    this_z1 = model.sample_z(imgs1.size(0))
    this_z2 = model.sample_z(imgs1.size(0))

    r1_z1 = model.gen.decode(labels1, this_z1)
    r1_z2 = model.gen.decode(labels1, this_z2)
    
    r_z0 = model.gen.decode(labels1, this_z1*0)

    r2_z1 = model.gen.decode(labels2, this_z1)
    r2_z2 = model.gen.decode(labels2, this_z2)

    mixes = [r1_z1, r1_z2, r_z0, r2_z1, r2_z2]

    mixes_labels = []
    mixes_labels.append("y1, z1")
    mixes_labels.append("y1, z2")
    mixes_labels.append("y1, 0")
    mixes_labels.append("y2, z1")
    mixes_labels.append("y2, z2")
        
    all_imgs = model.denorm(
        torch.cat([imgs1, 
                   imgs2,
                   imgs1*0] + \
                   mixes,
                   dim=0)
    ).cpu()
    
    im_size = imgs1.size(-1)
    pad_size = 2
    
    images = torchvision.utils.make_grid(
        all_imgs,
        nrow=imgs1.shape[0],
        pad_value=1
    )
    
    pil_img = _annotate_image(
        images,
        im_size=im_size,
        pad_size=pad_size,
        labels=["x1", "x2", ""] + \
            mixes_labels
    )
    pil_img.save(savedir)        

def viz_mixup_class_interpolation(model,
                                  labels1,
                                  labels2,
                                  im_size,
                                  savedir,
                                  **kwargs):
    
    gen = model.gen
    
    mixes = []
    mixes_labels = []
    bs = labels1.size(0)
    fixed_z = model.sample_z(bs)
    zeros = torch.zeros((labels1.size(0), 1)).to(model.rank)
    for alpha in np.linspace(0, 1, num=16):
        alpha_t = zeros + alpha
        this_mix = gen.decode_mixup(labels1,
                                    labels2,
                                    alpha_t,
                                    fixed_z)
        mixes.append(this_mix)
        mixes_labels.append("%.2f" % alpha)
        
    all_imgs = model.denorm(
        torch.cat(mixes, dim=0)
    ).cpu()

    pad_size = 2
    images = torchvision.utils.make_grid(
        all_imgs,
        nrow=bs,
        pad_value=1
    )
    
    pil_img = _annotate_image(
        images,
        im_size=im_size,
        pad_size=pad_size,
        labels=mixes_labels
    )
    pil_img.save(savedir)
    
def viz_reencoding(model,
                   imgs1,
                   labels1,
                   savedir,
                   **kwargs):
    
    #model.use_ema = False

    # TODO: be able to set the seed

    model.eval()
    
    # Use the same reference image for each column.
    this_r = model.gen.encode(imgs1)[0]
    this_z = torch.zeros_like(this_r).normal_(0,1)

    xfake = model.gen.decode(this_r, this_z)
    
    this_r_again = model.gen.encode(xfake)[0]
    
    xfake_again = model.gen.decode(this_r_again, this_z)

    mixes = [xfake, xfake_again]

    mixes_labels = []
    mixes_labels.append("rx(x1), z1")
    mixes_labels.append("rx(xf), z2")
        
    all_imgs = model.denorm(
        torch.cat([imgs1, 
                   imgs1*0] + \
                   mixes,
                   dim=0)
    ).cpu()
    
    im_size = imgs1.size(-1)
    pad_size = 2
    
    images = torchvision.utils.make_grid(
        all_imgs,
        nrow=imgs1.shape[0],
        pad_value=1
    )
    
    pil_img = _annotate_image(
        images,
        im_size=im_size,
        pad_size=pad_size,
        labels=["x1", ""] + \
            mixes_labels
    )
    pil_img.save(savedir)        


def reconstruct_for_all_labels(model, imgs, labels, all_labels, out_file):
    """Perform reconstruction on imgs for all labels, i.e.
    for all y produce rec(x|y).

    Args:
        model ([type]): the model
        imgs ([type]): the images
        labels ([type]): the labels corresponding to those images
        all_labels ([type]): a list of all valid labels
        out_file ([type]): output image

    Returns:
        [type]: the predicted labels per image
    """
    imgs = imgs[:,0].to(model.rank)
    all_labels = all_labels.to(model.rank)
    # (bs, n_labels, f, h, w)
    #imgs_reshp = imgs.unsqueeze(1).repeat(1, len(all_labels), 1, 1, 1)
    #imgs_reshp_flat = imgs_reshp.view(-1, imgs.size(1), imgs.size(2), imgs.size(3))

    # Do this inefficiently for now, one image at a time
    buf = []
    pred_labels = []
    for j in range(len(imgs)):
        this_img = imgs[j:j+1]
        this_img_repeat = this_img.repeat(len(all_labels), 1, 1, 1)
        imgs_recon, nlls = model.reconstruct(
            this_img_repeat.unsqueeze(1),
            all_labels,
            compute_nll=True
        )
        buf.append(imgs_recon.squeeze(1))
        # Evaluate the NLL on the entire batch
        label_argmin = torch.argmin(nlls)
        pred_label = all_labels[label_argmin]
        pred_labels.append(pred_label.item())

    pred_acc = (labels == torch.LongTensor(pred_labels)).float().mean()

    buf = torch.stack(buf)
    # buf shape = (bs, n_labels, 3, 32, 32)
    buf = buf.view(-1, buf.size(2), buf.size(3), buf.size(4))
    save_image(
        buf*0.5 + 0.5,
        out_file,
        nrow=len(all_labels)
    )

    return pred_acc

def sample(model,
           dataset,
           savedir,
           bs=16,
           **kwargs):
    
    #gen = model.gen
    # HACK
    #bs = labels.size(0)

    #labels = labels.unique()

    labels = torch.sort(dataset.unique_targets).values.to(model.rank)
    
    buf = []
    for j in range(len(labels)):
        label_repeat = labels[j:j+1].repeat(bs)
        this_samples = model.sample(label_repeat)
        buf.append(this_samples)
    
    # Use the same reference image for each column.
    all_imgs = torch.cat(buf, dim=0)
    
    save_image(
        all_imgs,
        savedir,
        nrow=bs,
        pad_value=1
    )