import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.training import main as train_main
from src.inference import main as inference_main
from src.utils.settings import Settings


def get_parser(main: bool = False) -> ArgumentParser:

    """
    Returns ArgumentParser object for main or scripts CLI entrypoints.

    Args:
        main (bool): Whether the parser is needed for main CLI entrypoint and should support mode choice.
    
    Returns:
        ArgumentParser: Parser for supported arguments and overrides.
    """

    mode_list = ['train', 'inference']

    parser = ArgumentParser()

    if main:
        parser.add_argument('command', choices=mode_list)

    parser.add_argument('--data', type=str, default=None, 
                        help='Path to images.')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to YAML config file.')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to model state dictionary.')
    parser.add_argument('--metadata', type=str, default=None, 
                        help='Path to metadata file with ground truth coordinates.')
    
    return parser



def cli_to_settings(cli_args: Namespace = None, cli_overrides: list[str] = None) -> Settings:

    """
    Builds Settings object out of CLI input and defaults.

    Args:
        cli_args: Named CLI arguments. Provide if it has already been parsed.
        cli_overrides: Unnamed CLI arguments. Provide if it has already been parsed.

    Returns:
        Settings: Built Settings object.

    Raises:
        ValueError: If unknown command is passed. Supports "train" and "inference".
    """
    
    if cli_args is None or cli_overrides is None:

        parser = get_parser()
        cli_args, cli_overrides = parser.parse_known_args()

    named_overrides = []
    mapping = {
        'command': 'mode',
        'data': 'data.path',
        'metadata': 'data.metadata_path',
        'model': 'model.path'
    }

    for arg_name, key in mapping.items():

        if hasattr(cli_args, arg_name):
            value = getattr(cli_args, arg_name, None)
            if value is not None:
                named_overrides.append(f'{key}={value}')

    overrides = cli_overrides + named_overrides

    config_path = Path(cli_args.config).resolve() if (cli_args.config is not None) else None

    settings = Settings.from_sources(
        config_path=config_path,
        overrides=overrides,
    )

    return settings


def main():
    
    parser = get_parser(main=True)
    cli_args, cli_overrides = parser.parse_known_args()

    command = cli_args.command

    settings = cli_to_settings(cli_args=cli_args, cli_overrides=cli_overrides)


    if command == 'train':
        train_main(settings=settings)

    elif command == 'inference':
        inference_main(settings=settings)


if __name__ == '__main__':

    main()
