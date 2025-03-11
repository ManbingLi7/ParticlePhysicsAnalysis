
import yaml

import os

def get_config(filename=None):
    if filename is None:
        filename = os.environ.get("ANTIMATTER_CONFIG", None)
    if filename is None:
        raise ValueError("$ANTIMATTER_CONFIG not set.")
    with open(filename) as config_file:
        return yaml.safe_load(config_file)


def get_default_rigidity_estimator(config):
    return config["analysis"]["rigidity_estimator"]

    
