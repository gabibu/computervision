import os

CONFIG_FILE_NAME = "config.yaml"


def validate_config_file_exist(exp_dir: str):
    config_file_name = [file for file in os.listdir(exp_dir) if file == CONFIG_FILE_NAME]

    if len(config_file_name) == 0 or len(config_file_name) > 1:
        raise ValueError(f"number of config file must be 1 ({len(config_file_name)})")


def get_weights_dir(experiment_dir: str, create_dir: bool = False):
    dir = os.path.join(experiment_dir, "weights")

    if create_dir:
        os.makedirs(dir, exist_ok=True)

    return dir


def get_weights_file_name(loss: float, epoc: int):
    return f"epoc_{epoc}_loss_{loss}.weights"


def get_config_file_path(exp_dir) -> str:
    return os.path.join(exp_dir, CONFIG_FILE_NAME)
