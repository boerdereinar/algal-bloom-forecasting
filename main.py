import inspect
import os
import sys
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter, RawTextHelpFormatter
from typing import List, Iterable, Tuple

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
    preprocessor_class = getattr(edegruyl.preprocessing, name)

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
    model_class = getattr(edegruyl.models, model_name)

    datamodule_name = f"{args.data_module}DataModule"
    datamodule_class = getattr(edegruyl.datamodules, datamodule_name)

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
    # Get all classes
    preprocessors = list(get_classes_in_module_endswith(edegruyl.preprocessing, "Preprocessor"))
    models = list(get_classes_in_module_endswith(edegruyl.models, "Classifier"))
    data_modules = list(get_classes_in_module_endswith(edegruyl.datamodules, "DataModule"))

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

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
    data_module_help = "The type of the datamodule to load the data with.\nAvailable data modules:\n" +\
                       os.linesep.join(f"\t{x:15}: {y}" for x, y in data_modules)

    train_parser = subparsers.add_parser("train", add_help=len(sys.argv) in [3, 4], help="Train a model on a dataset.",
                                         formatter_class=RawTextHelpFormatter)
    train_parser.add_argument("model", choices=[x for x, _ in models], metavar="MODELS", help=model_help)
    train_parser.add_argument("data_module", choices=[x for x, _ in data_modules], metavar="DATA_MODULES",
                              help=data_module_help)
    train_parser.set_defaults(func=train)

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
