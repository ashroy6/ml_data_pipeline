from pathlib import Path


def test_required_project_files_exist():
    required = [
        "Dockerfile",
        "requirements.txt",
        "src/train.py",
        "src/predict.py",
    ]
    for file_path in required:
        assert Path(file_path).exists(), f"Missing required file: {file_path}"
