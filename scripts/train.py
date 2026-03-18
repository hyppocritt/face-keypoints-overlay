from src.training import main as train_main
from src.cli import cli_to_settings

def main():

    settings = cli_to_settings()
    train_main(settings=settings)