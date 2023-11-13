import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import tss_lib.metric as module_metric
import tss_lib.model as module_model
from tss_lib.trainer import MetricTracker, Trainer
from tss_lib.utils import ROOT_PATH
from tss_lib.utils.object_loading import get_dataloaders
from tss_lib.config_processing.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model,)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]
    metrics_tracker = MetricTracker(*[met.name for met in metrics])

    # TODO: add option to save model's output audios

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            batch.update(output)
            for met in metrics:
                metrics_tracker.update(met.name, met(**batch))

    for metric_name in metrics_tracker.keys():
        avg_val = metrics_tracker.avg(metric_name)
        print(f'avg {metric_name}: \t{avg_val:.5f}')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="TSS testing script")
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
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            args_config = json.load(f)
            # override data
            if "data" in args_config:
                config.config["data"] = args_config["data"]
            # override postprocessor
            if "postprocessor" in args_config:
                config.config["postprocessor"] = args_config["postprocessor"]
            # override metrics
            if "metrics" in args_config:
                config.config["metrics"] = args_config["metrics"]
            config.config.update(args_config)

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirMixturesDataset",
                        "args": {
                            "mixtures_dir": str(test_data_folder)
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config)
