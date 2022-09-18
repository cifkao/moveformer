import contextlib
import os

import jsonargparse
from unittest import mock


@contextlib.contextmanager
def chdir(path):
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def load_experiment(cli_class, config_dict):
    parser = jsonargparse.capture_parser(cli_class)
    with mock.patch("sys.argv", ["", "fit"]):
        config = parser.parse_object({"fit": config_dict})
    objects = parser.instantiate_classes(config)["fit"]
    config = config["fit"]
    return config, objects
