import subprocess

def test_cli_override(data_path):

    result = subprocess.run(
        [
            'python', '-m', 'src',
            'inference',
            '--data', data_path,
            'inference.vis=first'
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0

if __name__ == '__main__':

    data_path = './tests/data/sample_image.jpg'

    test_cli_override(data_path=data_path)