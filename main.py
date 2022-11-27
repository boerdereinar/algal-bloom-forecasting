import sys
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from importlib import import_module
from pkgutil import iter_modules
from typing import List, Iterable

from pytorch_lightning import Trainer

import edegruyl.datamodules
import edegruyl.models
import edegruyl.preprocessing


def preprocess(args: Namespace, unknown_args: List[str]) -> None:
    """Runs the preprocessor.

    Args:
        args: The arguments from the parent parser. Must contain the `preprocessing` key.
        unknown_args: The remaining arguments.
    """
    name = f"{args.preprocessor}Preprocessor"
    preprocessor_class = getattr(import_module(f"edegruyl.preprocessing.{name}"), name)

    preprocessor_parser = ArgumentParser(prog=f"main.py preprocess {args.preprocessor}")
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
    model_name = f"{args.model}Classifier"
    model_class = getattr(import_module(f"edegruyl.models.{model_name}"), model_name)

    datamodule_name = f"{args.data_module}DataModule"
    datamodule_class = getattr(import_module(f"edegruyl.datamodules.{datamodule_name}"), datamodule_name)

    trainer_parser = ArgumentParser(prog=f"main.py train {args.model}", formatter_class=ArgumentDefaultsHelpFormatter)
    Trainer.add_argparse_args(trainer_parser)
    model_class.add_model_specific_args(trainer_parser)
    datamodule_class.add_datamodule_specific_args(trainer_parser)
    trainer_args = trainer_parser.parse_args(unknown_args)

    model = model_class(**vars(trainer_args))
    datamodule = datamodule_class(**vars(trainer_args))

    trainer: Trainer = Trainer.from_argparse_args(trainer_args)
    trainer.fit(model, datamodule)


def main() -> None:
    # Get all class names
    preprocessors = [submodule.name[:-12] for submodule in iter_modules(edegruyl.preprocessing.__path__)
                     if submodule.name.endswith("Preprocessor") and len(submodule.name) > 12]
    models = [submodule.name[:-10] for submodule in iter_modules(edegruyl.models.__path__)
              if submodule.name.endswith("Classifier") and len(submodule.name) > 10]
    data_modules = [submodule.name[:-10] for submodule in iter_modules(edegruyl.datamodules.__path__)
                    if submodule.name.endswith("DataModule") and len(submodule.name) > 10]

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", add_help=len(sys.argv) == 3,
                                              help="Preprocess the raw data.")
    preprocess_parser.add_argument("preprocessor", choices=preprocessors, help="The type of the preprocessor to use.")
    preprocess_parser.set_defaults(func=preprocess)

    # Train
    train_parser = subparsers.add_parser("train", add_help=len(sys.argv) in [3, 4], help="Train a model on a dataset.")
    train_parser.add_argument("model", choices=models, help="The type of the model to train on.")
    train_parser.add_argument("data_module", choices=data_modules,
                              help="The type of the datamodule to load the data with.")
    train_parser.set_defaults(func=train)

    # Run the parser
    args, unknown_args = parser.parse_known_args()
    args.func(args, unknown_args)


def get_classes_in_module_endswith(path: Iterable[str], suffix: str) -> Iterable[str]:
    """Gets all classes that end with a given suffix.

    Args:
        path: A list of paths to look for modules in.
        suffix: The suffix of the classes.

    Returns:
        All classes that end with the given suffix, with the suffix removed.
    """
    for submodule in iter_modules(path):
        if submodule.name.endswith(suffix) and len(submodule.name) > len(suffix):
            yield submodule.name[:-len(suffix)]


if __name__ == "__main__":
    main()
