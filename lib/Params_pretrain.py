import argparse
# import numpy as np
import configparser
# import pandas as pd

def parse_args(device):
    # parser
    args = argparse.ArgumentParser(prefix_chars='-', description='pretrain_arguments')
    args, _ = args.parse_known_args()
    return args