import os
from pathlib import Path

project_name = "loan_approval"

list_of_files = [
    ".github/workflows/.gitkeep",

    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    "notebooks/01_eda.ipynb",

    "artifacts/.gitkeep",
    "logs/.gitkeep",
    "tests/__init__.py",

    f"src/{project_name}/__init__.py",

    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",

    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/logger.py",

    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",

    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/exception/custom_exception.py",

    f"src/{project_name}/constants/__init__.py",

    "app.py",
    "setup.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    ".env",
    "config.yaml",
    "params.yaml",
    "dvc.yaml",
    "Dockerfile",
    ".dockerignore"
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir = filepath.parent
    os.makedirs(filedir, exist_ok=True)

    if not filepath.exists():
        filepath.touch()

print("Project structure created successfully!")