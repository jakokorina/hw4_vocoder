import argparse
import collections
import warnings

import numpy as np
import torch
from itertools import chain

import hw_tts.model as module_arch
from hw_tts.trainer import Trainer
from hw_tts.utils import prepare_device
from hw_tts.utils.object_loading import get_dataloaders
from hw_tts.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    mpd = config.init_obj(config["mpd"], module_arch)
    msd = config.init_obj(config["msd"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    mpd.to(device)
    msd.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params_d = filter(lambda p: p.requires_grad, chain(mpd.parameters(), msd.parameters()))
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    optimizer_d = config.init_obj(config["optimizer_d"], torch.optim, trainable_params_d)
    lr_scheduler_g = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)
    lr_scheduler_d = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_d)

    trainer = Trainer(
        model,
        mpd,
        msd,
        optimizer,
        optimizer_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
        CustomArgs(
            ["--ne", "--n_epoch"], type=int, target="trainer;epochs"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
