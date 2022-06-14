'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

from datetime import datetime
import argparse
import json
import os
import re


class Settings(object):

    def __init__(
            self,
            config_path='config.json',
            state_path='state.json',
            info_path='info.md'
    ):

        self.config_path = config_path
        self.state_path = state_path
        self.info_path = info_path


DEFAULT_SETTINGS = Settings()


class Option(object):

    def __init__(
            self,
            name=None,
            type=None,
            required=True,
            value=None,
            load_value=None,
            command_line=True,
            nargs=None,
            check_type=True,
            help='',
    ):
        if nargs == '?':
            raise ValueError('nargs == \'?\' is not supported')

        if nargs is not None and type == bool:
            raise ValueError(
                'nargs is not None and type == bool is not supported')

        self.name = name
        self.type = type
        self.required = required
        self.value = value
        self.load_value = load_value
        self.command_line = command_line
        self.nargs = nargs
        self.check_type = check_type
        self.help = help


class Config(object):

    @staticmethod
    def serialize(config):
        return config.__dict__.copy()


class ConfigBuilder(object):

    def __init__(self, options, load=False):
        self.options = self.build_options_dict(options)
        self.config = Config()
        self.initialize_config(self.options, load)

    def initialize_config(self, options, load=False):
        for name, option in options.items():
            self.apply_entry(name, option.load_value if (load and hasattr(option, 'load_value'))else option.value)

    def get_built_in_options(self):
        return [
            Option(
                name='debug',
                type=bool,
                value=False,
                help='Enable debug mode',
            ),
            Option(
                name='scratch',
                type=bool,
                value=False,
                help='Enable scratch mode (save results to scratch folder instead of as experiment)',
            ),
            Option(
                name='experiment_path',
                type=str,
                required=False,
                help='Path to experiment folder',
            ),
            Option(
                name='experiment_id',
                type=str,
                required=False,
                help='Short ID of experiment',
            ),
        ]

    def build_options_dict(self, options):
        options_dict = {}

        for option in self.get_built_in_options():
            options_dict[option.name] = option

        if isinstance(options, dict):
            for name, option in options.items():
                option.name = name
                options_dict[name] = option

        elif isinstance(options, list):
            for option in options:
                if option.name in options_dict:
                    raise ValueError(
                        'Option "{}" already defined'.format(option.name))

                options_dict[option.name] = option

        else:
            raise ValueError(
                'Options must be dict or list, but got "{}"'.format(options))

        return options_dict

    def get_arguments(self):
        arguments = {}

        def add_arg(arg_key, params):
            if arg_key in arguments:
                raise ValueError(
                    'Command line argument "{}" already defined'.format(arg_key))

            arguments[arg_key] = params

        for key, option in self.options.items():
            if not option.command_line:
                continue

            if option.type == bool:
                add_arg(
                    key,
                    {
                        'action': 'store_const',
                        'const': True,
                        'required': False,
                        'help': option.help,
                    }
                )

                add_arg(
                    'no_' + key,
                    {
                        'action': 'store_const',
                        'const': False,
                        'required': False,
                        'dest': key,
                        'help': '(Disable) ' + option.help,
                    }
                )

            else:
                params = {
                    'type': option.type,
                    'required': False,
                    'default': option.value,
                    'help': option.help,
                }

                if option.nargs is not None:
                    params['nargs'] = option.nargs

                add_arg(key, params)

        return arguments

    def apply_entry(self, key, value):

        if key not in self.options:
            raise ValueError('Option "{}" not defined'.format(key))

        option = self.options[key]
        type = option.type

        if option.check_type and type is not None and value is not None:
            if type == float:
                type = (int, float)

            if option.nargs is not None:
                if not isinstance(value, list):
                    raise ValueError('Option "{}" must be a list, but given "{}" instead'.format(
                        key, value))

                for arg in value:
                    if not isinstance(arg, type):
                        raise ValueError('Element in option "{}" must be of type "{}", but given "{}" instead'.format(
                            key, option.type.__name__, arg))

            else:
                if not isinstance(value, type):
                    raise ValueError('Option "{}" must be of type "{}", but given "{}" instead'.format(
                        key, option.type.__name__, value))

        setattr(self.config, key, value)

    def apply_dict(self, obj):
        # import ipdb
        # ipdb.set_trace()
        for key, value in obj.items():
            self.apply_entry(key, value)

    def apply_args(self, args):
        for name, option in self.options.items():
            if not option.command_line:
                continue

            value = getattr(args, name, None)
            if value is not None:
                self.apply_entry(name, value)

    def validate(self):
        for name, option in self.options.items():
            if option.required and getattr(self.config, name, None) is None:
                raise ValueError(
                    'Option "{}" is required but not specified'.format(name))

    def build(self):
        self.validate()
        return self.config


class State(object):

    def __init__(self):
        self.start_time = datetime.now()
        self.reset()

    def reset(self):
        self.started = False
        self.completed = False

    @staticmethod
    def serialize(state):
        data = state.__dict__.copy()
        data['start_time'] = state.start_time.timestamp()
        return data

    @staticmethod
    def deserialize(data):
        state = State()
        state.__dict__.update(data)
        if 'start_time' in data:
            state.start_time = datetime.fromtimestamp(data['start_time'])

        else:
            print('Warning: no start_time found in state. Using current time.')
            state.start_time = datetime.now()

        return state


class Experiment(object):

    def __init__(self, root_path, config, state, settings):
        self.root_path = root_path
        self.config = config
        self.state = state
        self.settings = settings

    @staticmethod
    def new(default_folder_path=None,
            short_info='(No short info)',
            info='(No info)',
            options=[],
            extra_config=None,
            no_output=False,
            settings=DEFAULT_SETTINGS):

        builder = ConfigBuilder(options)

        if extra_config is not None:
            builder.apply_dict(extra_config)

        arguments = builder.get_arguments()
        args = Experiment.parse_args(arguments, short_info)
        builder.apply_args(args)

        config = builder.build()

        state = State()

        if no_output:
            root_path = None

        else:
            if not default_folder_path:
                raise ValueError('Default folder path is not specified')

            if config.scratch:
                if config.experiment_id:
                    raise ValueError(
                        '--scratch is true, therefore --experiment_id should not be given')

                print(
                    'NOTE: --scratch mode is enabled: This experiment is not permanently saved.')

            else:
                if not config.experiment_id:
                    raise ValueError(
                        'Please provide an --experiment_id (or specify to use --scratch mode)')

                if re.match(r'^[a-zA-Z0-9_\-]+$',
                            config.experiment_id) is None:
                    raise ValueError('Invalid experiment id "{}"'.format(
                        config.experiment_id))

            root_path = Experiment.get_root_path(
                default_folder_path, config, state)

        experiment = Experiment(
            root_path=root_path,
            config=config,
            state=state,
            settings=settings
        )

        if not no_output:
            experiment.save_config(settings.config_path)
            experiment.save_state(settings.state_path)
            experiment.save_info(settings.info_path,
                                 short_info,
                                 info)

        return experiment

    @staticmethod
    def load(experiment_path,
             options=[],
             settings=DEFAULT_SETTINGS):

        print('Loading experiment from {}'.format(experiment_path))

        with open(os.path.join(experiment_path,
                               settings.config_path),
                  'r') as file:
            builder = ConfigBuilder(options, load=True)
            builder.apply_dict(json.load(file))
            config = builder.build()

        with open(os.path.join(experiment_path,
                               settings.state_path),
                  'r') as file:
            state = State.deserialize(json.load(file))

        experiment = Experiment(
            root_path=experiment_path,
            config=config,
            state=state,
            settings=settings,
        )

        return experiment

    @staticmethod
    def get_root_path(default_folder_path, config, state):
        folder_path = config.experiment_path
        if not folder_path:
            folder_path = default_folder_path

        if config.scratch:
            return os.path.join(
                folder_path,
                'scratch_2',
            )

        return os.path.join(
            folder_path,
            '{}_{}'.format(
                state.start_time.strftime('%Y-%m-%d_%H-%M-%S'),
                config.experiment_id,
            ),
        )

    @staticmethod
    def parse_args(arguments, short_info):
        parser = argparse.ArgumentParser(
            description=short_info)

        for name in arguments:
            parser.add_argument('--' + name,
                                **(arguments[name]))

        return parser.parse_args()

    def file_path(self, path, ensure_parent=True):
        # if not self.started:
        #     raise ValueError(
        #         'Cannot call Experiment.path without running the experiment')

        full_path = os.path.join(self.root_path, path)
        if ensure_parent:
            self.ensure_parent_dir(full_path)

        return full_path

    def dir_path(self, path, ensure=True):
        # if not self.started:
        #     raise ValueError(
        #         'Cannot call Experiment.path without running the experiment')

        full_path = os.path.join(self.root_path, path)
        if ensure:
            self.ensure_full_path(full_path)

        return full_path

    def ensure_full_path(self, full_path):
        os.makedirs(full_path, exist_ok=True)

    def ensure_dir_path(self, path):
        self.dir_path(path, ensure=True)

    def ensure_parent_dir(self, full_path):
        self.ensure_full_path(os.path.dirname(full_path))

    def open(self, path, *args, **kwargs):
        full_path = self.file_path(path)
        self.ensure_parent_dir(full_path)
        return open(full_path, *args, **kwargs)

    def save_text(self, path, text):
        with self.open(path, 'w') as file:
            file.write(text)

    def save_config(self, path):
        with self.open(path, 'w') as file:
            json.dump(Config.serialize(self.config),
                      file, indent=2, sort_keys=True)

    def save_state(self, path):
        with self.open(path, 'w') as file:
            json.dump(State.serialize(self.state),
                      file, indent=2, sort_keys=True)

    def get_title(self):
        return '{} {}'.format(
            self.state.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            self.config.experiment_id if self.config.experiment_id
            else '(No name)',
        )

    def save_info(self, path, short_info, info):
        self.save_text(path, '''# Experiment {}
{}

----------

{}
'''.format(self.get_title(), short_info, info))

    def on_start(self):
        self.state.reset()
        self.state.started = True

        print('{}'.format(self.get_title()))
        self.print_config(self.config)

        if self.root_path:
            self.save_config(self.settings.config_path)
            self.save_state(self.settings.state_path)

    def on_complete(self):
        self.state.completed = True

        if self.root_path:
            self.save_state(self.settings.state_path)

    def print_config(self, config):
        from pprint import pprint
        pprint(config.__dict__)

    def run(self, func):
        self.on_start()
        func(self, self.config, self.state)
        self.on_complete()


###########################
# Example usage
###########################

if __name__ == '__main__':

    ###########################
    # Perform an experiment
    ###########################

    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

    SHORT_INFO = 'My Experiment'
    INFO = 'Longer description of the experiment'
    DEFAULT_FOLDER_PATH = os.path.join(ROOT_DIR, 'experiments')

    # Each experiment ran will be saved to its own folder, under the
    # DEFAULT_FOLDER_PATH.

    # A few things will be saved into the experiment folder by default:
    # - The description (SHORT_INFO and INFO) of the experiment
    # - The config options used for the experiment
    #   Will be saved to a human-readable JSON
    # - The state of the experiment (e.g. completed or not)

    # Command line arguments are generated from the options.
    # Options can be altered from the command line, such as using
    # `--dataset_path /path/to/my/dataset`.

    # A few built-in options are already set:
    # --experiment_id: REQUIRED. The short identifier of the experiment.
    # --debug: Whether to enable debug mode. User chooses what to do with it.
    # --experiment_path: Path to the folder containing each experiment.

    # Note that an array of options would also work: pass `name='...'` to the
    # Option constructor.

    OPTIONS = {
        'dataset_path': Option(
            type=str,
            required=True,
            help='Path to dataset',
        ),
        'num_epochs': Option(
            type=int,
            value=1000,
            help='Number of epochs to train',
        ),
        'learning_rate': Option(
            type=float,
            value=0.0001,
        ),
        'enable_discriminator': Option(
            # For command line, use `--enable_discriminator` for True,
            # and `--no_enable_discriminator` for False.
            type=bool,
            value=True,
        ),
    }

    experiment = Experiment.new(
        default_folder_path=DEFAULT_FOLDER_PATH,
        short_info=SHORT_INFO,
        info=INFO,
        options=OPTIONS,
    )

    def main(experiment, config, state):
        # Obtain file path relative to experiment folder.
        # By default, `ensure_parent=True`, which ensures that the directory of
        # given file path is created if not already exists.
        print(experiment.file_path('checkpoint.pth'))

        # Obtain directory path relative to experiment folder.
        # By default, `ensure=True`, which ensures that the given directory is
        # created if not # already exists.
        print(experiment.dir_path('samples'))

        # Access config options like object attributes
        for epoch in range(config.num_epochs):
            print(epoch)

    experiment.run(main)

    # The minimal command line usage of this script will be:
    #     python my_experiment.py --experiment_id <id>
    # Where <id> is a small identifier given to the experiment, such as
    # "test-dropout".

    # This will save the config options and other information into
    # experiments/2020-02-03_14-17-14_test-dropout.

    ###########################
    # Loading an experiment
    ###########################

    # To load a previously ran experiment, do the following:

    experiment_path = "experiments/2020-02-03_14-17-14_test-dropout"

    experiment = Experiment.load(
        experiment_path
    )

    # Access config options like so:

    print(experiment.config.enable_discriminator)

    # Obtain path relative to experiment folder also works:

    print(experiment.file_path('checkpoint.pth'))
    print(experiment.dir_path('samples'))
