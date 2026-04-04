from src.cli import cli_to_settings
from src.training import main as train_main


def main():

    settings = cli_to_settings()
    train_main(settings=settings)
