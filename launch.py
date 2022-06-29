import argparse
import os
from importlib import import_module
from src import setup_logger
from exp_configs import exp_utils
import pprint

logger = setup_logger.get_logger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #parser.add_argument(
    #    "-e", "--exp_group_list", nargs="+", help="Define which exp groups to run."
    #)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json_file", default=None)
    group.add_argument("--cfg_file", default=None)
    parser.add_argument(
        "--tl",
        "--task_launcher",
        default=None,
        required=True,
        help="Define python file which contains trainval method, e.g. trainval.py",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action='store_true',
        help="If set, run the specified trainval module but exit before training starts"
    )
    parser.add_argument(
        "-s",
        "--savedir",
        default=None,
        required=True,
        help="Define the base directory where the experiment will be saved.",
    )
    parser.add_argument(
        "-d", "--datadir", default=None, help="Define the dataset directory."
    )
    parser.add_argument(
        "-nw", "--num_workers", default=8, type=int, help="num_workers"
    )
    #parser.add_argument(
    #    "-r",
    #    "--resume",
    #    action='store_true',
    #    help="""Are we resuming an experiment? If this is set to True,
    #    then the specified task launcher will not overwrite exp_dict.json
    #    which resides in `savedir`. 
    #    """
    #)
    
    args, others = parser.parse_known_args()
    logger.info(args)

    if args.tl is not None and args.tl.endswith(".py"):
        args.tl = args.tl.replace(".py", "")

    this_module = import_module(args.tl)
    logger.info(this_module)

    if not os.path.exists(args.savedir):
        logger.info("{} does not exist so creating...".format(args.savedir))
        os.makedirs(args.savedir)

    if args.json_file is not None:
        configs = exp_utils.enumerate_and_unflatten(args.json_file)

        if len(configs) > 1:
            logger.warning(
                "More than one configuration detected in json file, "
                "automatically selecting the first since this script does "
                "not support launching multiple experiments at once."
            )
        exp_dict = configs[0]
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("\n" + pp.pformat(exp_dict))
    else:
        raise NotImplementedError("TODO")

    if not hasattr(this_module, "trainval"):
        raise NotImplementedError(
            "module {} must have a trainval(exp_dict, savedir, args) method".format(
                args.tl
            )
        )

    this_module.trainval(exp_dict, args.savedir, args)
