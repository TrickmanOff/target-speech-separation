import argparse
import json

import tss_lib.storage as storage_module
from tss_lib.storage.experiments_storage import ExperimentsStorage
from tss_lib.storage.external_storage import ExternalStorage
from tss_lib.config_processing.parse_config import ConfigParser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="external storage config file path (default: None)",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="saved/models",
        type=str,
        help="directory where the models are stored",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=str,
        help="run to load in format '{exp_name}:{run_name}'",
    )
    parser.add_argument(
        "command",
        type=str,
        nargs='*',
        help="'list' or 'checkpoint checkpoint_name' or 'config'"
    )

    args = parser.parse_args()

    config_filepath = args.config
    with open(config_filepath, 'r') as file:
        config = json.load(file)

    external_storage: ExternalStorage = ConfigParser.init_obj(config["external_storage"], storage_module)
    
    exps_storage = ExperimentsStorage(args.path)
    exp_name, run_name = args.run.split(':')
    run_storage = exps_storage.get_run(exp_name, run_name, create_run_if_no=True)
    if len(args.command) == 1 and args.command[0] == 'list':
        print(external_storage.list_content())
    elif len(args.command) == 1 and args.command[0] == 'config':
        print(f'Loading config for run {run_storage.get_run_id()}...')
        external_storage.import_config(run_storage)
        print(f'Successfully loaded config for run "{run_storage.get_run_id()}"')
    elif len(args.command) == 2 and args.command[0] == 'checkpoint':
        checkpoint_name = args.command[1]
        print(f'Loading checkpoint "{checkpoint_name}" for run {run_storage.get_run_id()}...')
        external_storage.import_checkpoint(run_storage, checkpoint_name)
        print(f'Successfully loaded checkpoint "{checkpoint_name}" for run "{run_storage.get_run_id()}"')
    else:
        raise RuntimeError('Wrong command format')
