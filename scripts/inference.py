from src.cli import cli_to_settings
from src.inference import main as inference_main


def main():

    settings = cli_to_settings()
    inference_main(settings=settings)
