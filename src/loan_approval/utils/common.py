import yaml


def read_yaml(path_to_yaml: str) -> dict:
    """
    Read YAML file and return dictionary.
    """

    with open(path_to_yaml, "r") as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content
import os
import yaml
import joblib


def read_yaml(path_to_yaml: str) -> dict:

    with open(path_to_yaml, "r") as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content


def save_object(file_path, obj):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    joblib.dump(obj, file_path)

