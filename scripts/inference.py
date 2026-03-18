from src.inference import main as inference_main
from src.cli import cli_to_settings

def main():

    settings = cli_to_settings()
    inference_main(settings=settings)