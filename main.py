import inspect
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace, RawTextHelpFormatter
from typing import Iterable, List, Tuple

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

import edegruyl.models
import edegruyl.preprocessing
from edegruyl.analysis import Analyser
from edegruyl.datamodules import RioNegroDataModule

# Increase the timeout duration of the _wait_for_ports function from 30 seconds to 300 seconds.
# This patch fixes wandb failing to find ports on a slow cluster.
if "SLURM_JOB_ID" in os.environ:
    import subprocess
    import wandb.sdk.service.service

    def _wait_for_ports_decorator(original_method):
        def _wait_for_ports(self, fname: str, proc: subprocess.Popen = None) -> bool:
            return any(original_method(self, fname, proc) for _ in range(10))
        return _wait_for_ports

    wandb.sdk.service.service._Service._wait_for_ports = \
        _wait_for_ports_decorator(wandb.sdk.service.service._Service._wait_for_ports)


def analyse(args: Namespace, unknown_args: List[str]) -> None:
    """Runs the analyzer.

    Args:
        args: The arguments from the parent parser.
        unknown_args: The remaining arguments.
    """
    analyser = Analyser(**vars(args))
    analyser.analyse()


def preprocess(args: Namespace, unknown_args: List[str]) -> None:
    """Runs the preprocessor.

    Args:
        args: The arguments from the parent parser. Must contain the `preprocessing` key.
        unknown_args: The remaining arguments.
    """
    name = f"{args.preprocessor}Preprocessor"
    preprocessor_class = getattr(edegruyl.preprocessing, name)

    preprocessor_parser = ArgumentParser(prog=f"main.py preprocess {args.preprocessor}",
                                         formatter_class=RawTextHelpFormatter)
    preprocessor_class.add_preprocessor_specific_args(preprocessor_parser)
    preprocessor_args = preprocessor_parser.parse_args(unknown_args)

    preprocessor = preprocessor_class(**vars(preprocessor_args))
    preprocessor.preprocess()


def train(args: Namespace, unknown_args: List[str]) -> None:
    """Runs the training for a specific model.

    Args:
        args: The arguments from the parent parser. Must contain the `model` key.
        unknown_args: The remaining arguments.
    """
    model_name = f"{args.model}Model"
    model_class = getattr(edegruyl.models, model_name)

    trainer_parser = ArgumentParser(
        prog=f"main.py train {args.model}",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    Trainer.add_argparse_args(trainer_parser)
    model_class.add_model_specific_args(trainer_parser)
    RioNegroDataModule.add_datamodule_specific_args(trainer_parser)
    trainer_parser.add_argument("--disable-logger", action="store_true", help="Whether to disable the wandb logger.")
    trainer_args = trainer_parser.parse_args(unknown_args)

    # Model
    model = model_class(**vars(trainer_args), num_bands=3)
    datamodule = RioNegroDataModule(**vars(trainer_args))

    # Logging
    save_dir = trainer_args.default_root_dir or "./checkpoints"

    lr_monitor = LearningRateMonitor("step", True)
    logger = trainer_args.disable_logger or WandbLogger(project="algal-bloom", log_model=True, save_dir=save_dir)

    # Trainer
    trainer: Trainer = Trainer.from_argparse_args(
        trainer_args,
        callbacks=[lr_monitor],
        logger=logger
    )
    trainer.fit(model, datamodule)


def test(args: Namespace, unknown_args: List[str]) -> None:
    model_name = f"{args.model}Model"
    model_class = getattr(edegruyl.models, model_name)

    test_parser = ArgumentParser(
        prog=f"main.py test {args.model}",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    Trainer.add_argparse_args(test_parser)
    test_parser.add_argument("--checkpoint-path", type=str, help="The path to the checkpoint.", required=True)
    test_parser.add_argument("--save-dir", type=str, help="The save directory for the plots.", default=None)
    test_args = test_parser.parse_args(unknown_args)

    # Model
    model = model_class.load_from_checkpoint(test_args.checkpoint_path, save_dir=test_args.save_dir)
    datamodule = RioNegroDataModule.load_from_checkpoint(test_args.checkpoint_path)

    # Trainer
    trainer: Trainer = Trainer.from_argparse_args(test_args)

    trainer.test(model, datamodule)


def main() -> None:
    # Get all classes
    preprocessors = list(get_classes_in_module_endswith(edegruyl.preprocessing, "Preprocessor"))
    models = list(get_classes_in_module_endswith(edegruyl.models, "Model"))

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # Analyse
    analyse_parser = subparsers.add_parser("analyse", help="Analyze the data.")
    Analyser.add_analyser_specific_args(analyse_parser)
    analyse_parser.set_defaults(func=analyse)

    # Preprocess
    preprocessor_help = "The type of the preprocessor to use.\nAvailable preprocessors:\n" +\
                        os.linesep.join(f"\t{x:15}: {y}" for x, y in preprocessors)

    preprocess_parser = subparsers.add_parser("preprocess", add_help=len(sys.argv) == 3,
                                              help="Preprocess the raw data.", formatter_class=RawTextHelpFormatter)
    preprocess_parser.add_argument("preprocessor", choices=[x for x, _ in preprocessors], metavar="PREPROCESSORS",
                                   help=preprocessor_help)
    preprocess_parser.set_defaults(func=preprocess)

    # Train
    model_help = "The type of the model to train on.\nAvailable models:\n" +\
                 os.linesep.join(f"\t{x:15}: {y}" for x, y in models)

    train_parser = subparsers.add_parser("train", add_help=len(sys.argv) == 3, help="Train a model on a dataset.",
                                         formatter_class=RawTextHelpFormatter)
    train_parser.add_argument("model", choices=[x for x, _ in models], metavar="MODELS", help=model_help)
    train_parser.set_defaults(func=train)

    # Test
    model_help = model_help.replace("train on", "test")

    test_parser = subparsers.add_parser("test", add_help=len(sys.argv) == 3, help="Test a model on a dataset.",
                                        formatter_class=RawTextHelpFormatter)
    test_parser.add_argument("model", choices=[x for x, _ in models], metavar="MODELS", help=model_help)
    test_parser.set_defaults(func=test)

    # Run the parser
    args, unknown_args = parser.parse_known_args()
    args.func(args, unknown_args)


def get_classes_in_module_endswith(module: object, suffix: str) -> Iterable[Tuple[str, str]]:
    """Gets all classes that end with a given suffix.

    Args:
        module: The module.
        suffix: The suffix of the classes.

    Returns:
        All classes that end with the given suffix, with the suffix removed and the documentation of the class.
    """
    for name, klass in inspect.getmembers(module, inspect.isclass):
        if name.endswith(suffix) and len(name) > len(suffix):
            yield name[:-len(suffix)], klass.__doc__


if __name__ == "__main__":
    main()
