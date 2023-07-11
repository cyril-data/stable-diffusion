import argparse
import os
import sys
import glob
import datetime
import yaml
import torch
import time
import numpy as np
from tqdm import trange
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# makedirs_origin = os.makedirs
# def makedirs_pathlib(path, mode=0o777, exist_ok=False):
#     p = Path(path)
#     try:
#         p.mkdir(mode=mode, exist_ok=exist_ok)
#     except FileNotFoundError as e:
#         print(f"WARNING : {e} \n=> Nested directory creation activates")
#         p.mkdir(parents=True, exist_ok=exist_ok)
#     except FileExistsError as e:
#         print(f"WARNING : {e} \n=> the nested directory already exist")
#         pass
#     pass
# os.makedirs = makedirs_pathlib


def rescale(x): return (x + 1.) / 2.


def save_gray_image(grid, outfile, colormap):
    plt.imshow(grid, cmap=colormap)
    plt.colorbar()
    plt.savefig(outfile)
    plt.close()


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):

    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(
        steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):

    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log


def run(
        model, logdir, batch_size=50, vanilla=False, custom_steps=None,
        eta=None, n_samples=50000, nplog=None, gens=False, means=None, stds=None, png=False):

    if vanilla:
        print(
            f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(
            f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png')))
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(
                logs, logdir, n_saved=n_saved, key="sample", gens=gens, means=means, stds=stds, nplog=nplog, png=png)

            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]

    else:
        raise NotImplementedError(
            'Currently only sampling for unconditional models supported.')

    print(
        f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(
        logs, path, n_saved=0, key="sample", np_path=None, gens=False, means=None, stds=None, nplog=None, png=False):

    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    if gens:

                        normTransForPostGens = transforms.Compose([
                            transforms.Normalize(
                                mean=[0.] * 3, std=[1 / el for el in [2, 2, 2]]),
                            transforms.Normalize(
                                mean=[-el for el in [-1, -1, -1]], std=[1.] * 3),
                        ])
                        grid = normTransForPostGens(x)
                        grid = grid.detach().cpu()
                        grid = grid.numpy()

                        np.save(os.path.join(
                            nplog, f"_Fsample_0_{n_saved:06}.npy"), grid)

                        if png:
                            invTrans = transforms.Compose([
                                transforms.Normalize(
                                    mean=[0.] * 3, std=[1 / el for el in [2, 2, 2]]),
                                transforms.Normalize(
                                    mean=[-el for el in [-1, -1, -1]], std=[1.] * 3),
                                transforms.Normalize(
                                    mean=[0.] * 3, std=[1 / el for el in stds]),
                                transforms.Normalize(
                                    mean=[-el for el in means], std=[1.] * 3),
                            ])
                            grid = invTrans(x)
                            grid = torch.transpose(grid, 0, 2)
                            grid = torch.fliplr(grid)
                            grid = torch.rot90(grid, 2)
                            grid = grid.detach().cpu()
                            grid = grid.numpy()

                            os.makedirs(os.path.join(path, "u"), exist_ok=True)
                            os.makedirs(os.path.join(path, "v"), exist_ok=True)
                            os.makedirs(os.path.join(path, "t"), exist_ok=True)

                            save_gray_image(grid[:, :, 0], os.path.join(
                                os.path.join(path, "u"), f"u_{n_saved:06}.png"), 'viridis')
                            save_gray_image(grid[:, :, 1], os.path.join(
                                os.path.join(path, "v"), f"v_{n_saved:06}.png"), 'viridis')
                            save_gray_image(grid[:, :, 2], os.path.join(
                                os.path.join(path, "t"), f"t2m_{n_saved:06}.png"), 'RdBu_r')
                    else:
                        img = custom_to_pil(x)
                        print("save images path :", path)
                        print("save images key :", key)
                        print(f"save images n_saved:06 :  {n_saved:06}")
                        print(f" key + n_saved :      {key}_{n_saved:06}.png")
                        imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                        img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(
                    np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=''
    )

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-g",
        "--gens",
        help="param to save a images of type gens (t2m, u, v) ",
        default=True,
    )

    parser.add_argument(
        "--png",
        action='store_true'
    )

    return parser


def load_model_from_config(config, sd, gpus=False):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)

    if gpus:
        model = nn.DataParallel(model)
        model.to(torch.device(f"cuda:{gpus}"))
    else:
        model.to(torch.device("cpu"))

    model.eval()
    return model


def load_model(config, ckpt, eval_mode, gpus=False):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None

    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpus=gpus)

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    print("opt.base", opt.base)
    if len(opt.base) == 0:
        base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
        print("base_configs", base_configs)
        opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpus = opt.gpus if not opt.gpus == '' else False
    print('*'*80)
    print("gpus", gpus)
    print('*'*80)

    eval_mode = True
    gens = opt.gens

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "":
            locallog = logdir.split(os.sep)[-2]
        print(
            f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    # data

    Means = np.load("data/train_IS_1_1.0_0_0_0_0_0_256_done_red/" +
                    'mean_with_orog.npy')[[1, 2, 3]]
    Maxs = np.load("data/train_IS_1_1.0_0_0_0_0_0_256_done_red/" +
                   'max_with_orog.npy')[[1, 2, 3]]
    means = list(tuple(Means))
    stds = list(tuple((1.0/0.95)*(Maxs)))

    print("means", means)
    print("stds", stds)
    print("ckpt", ckpt)
    print("eval_mode", eval_mode)
    model, global_step = load_model(config, ckpt, eval_mode, gpus=gpus)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    run(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir, gens=gens, means=means, stds=stds, png=opt.png)

    print("done.")
