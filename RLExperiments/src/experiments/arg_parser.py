import configparser
import argparse


parser = argparse.ArgumentParser(description='Generate a runtime configuration for experiment.')

parser.add_argument('config_file', metavar='FILE')
parser.add_argument('--load_params', action='store_true')
parser.add_argument('--load_location', default='data/models/default.ckpt')


def parse_args(args):
    args = parser.parse_args(args[1:])
    return args


def parse_conf_file(conf_file):
    config = configparser.ConfigParser()
    config.read(conf_file)
    return config


def cast_conf_split(conf, cast=lambda x: x):
    return [cast(x) for x in conf.split(",")]

