from src.inference import run_inference


def test_inference(data_path):

    results = run_inference(
        data_path=data_path,
        vis='first',
        save=False
    )

    assert isinstance(results, dict)


if __name__ == '__main__':

    data_path = ...
    test_inference(data_path)
