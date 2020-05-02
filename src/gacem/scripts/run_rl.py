import spinup
from spinup.utils.run_utils import ExperimentGrid
from spinup import vpg_pytorch
from spinup.utils.serialization_utils import convert_json
import argparse
import gym
import json
import os, subprocess, sys
import os.path as osp
import string
import tensorflow as tf
import torch
from copy import deepcopy


import importlib
from textwrap import dedent

from spinup.run import (
    friendly_err,
)

from utils.pdb import register_pdb_hook

register_pdb_hook()


# Command line args that will go to ExperimentGrid.run, and must possess unique
# values (therefore must be treated separately).
RUN_KEYS = ['num_cpu', 'data_dir', 'datestamp']

# Command line sweetener, allowing short-form flags for common, longer flags.
SUBSTITUTIONS = {'env': 'env_name',
                 'dim': 'env_kwargs:dim',
                 'fn': 'env_kwargs:fn',
                 'hid': 'ac_kwargs:hidden_sizes',
                 'act': 'ac_kwargs:activation',
                 'cpu': 'num_cpu',
                 'dt': 'datestamp'}

# Only some algorithms can be parallelized (have num_cpu > 1):
MPI_COMPATIBLE_ALGOS = ['reinforce']

# Algo names (used in a few places)
BASE_ALGO_NAMES = ['reinforce']


def parse_and_execute_grid_search(cmd, args):
    """Interprets algorithm name and cmd line args into an ExperimentGrid."""

    algo = getattr(importlib.import_module(f'gacem.alg.{cmd}.{cmd}'), cmd)

    # Before all else, check to see if any of the flags is 'help'.
    valid_help = ['--help', '-h', 'help']
    if any([arg in valid_help for arg in args]):
        print('\n\nShowing docstring for gacem.alg.'+cmd+':\n')
        print(algo.__doc__)
        sys.exit()

    def process(arg):
        # Process an arg by eval-ing it, so users can specify more
        # than just strings at the command line (eg allows for
        # users to give functions as args).
        try:
            return eval(arg)
        except:
            return arg

    # Make first pass through args to build base arg_dict. Anything
    # with a '--' in front of it is an argument flag and everything after,
    # until the next flag, is a possible value.
    arg_dict = dict()
    for i, arg in enumerate(args):
        assert i > 0 or '--' in arg, \
            friendly_err("You didn't specify a first flag.")
        if '--' in arg:
            arg_key = arg.lstrip('-')
            arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(process(arg))


    # Make second pass through, to catch flags that have no vals.
    # Assume such flags indicate that a boolean parameter should have
    # value True.
    for k,v in arg_dict.items():
        if len(v) == 0:
            v.append(True)

    # Third pass: check for user-supplied shorthands, where a key has
    # the form --keyname[kn]. The thing in brackets, 'kn', is the
    # shorthand. NOTE: modifying a dict while looping through its
    # contents is dangerous, and breaks in 3.6+. We loop over a fixed list
    # of keys to avoid this issue.
    given_shorthands = dict()
    fixed_keys = list(arg_dict.keys())
    for k in fixed_keys:
        p1, p2 = k.find('['), k.find(']')
        if p1 >= 0 and p2 >= 0:
            # Both '[' and ']' found, so shorthand has been given
            k_new = k[:p1]
            shorthand = k[p1+1:p2]
            given_shorthands[k_new] = shorthand
            arg_dict[k_new] = arg_dict[k]
            del arg_dict[k]

    # Penultimate pass: sugar. Allow some special shortcuts in arg naming,
    # eg treat "env" the same as "env_name". This is super specific
    # to Spinning Up implementations, and may be hard to maintain.
    # These special shortcuts are described by SUBSTITUTIONS.
    for special_name, true_name in SUBSTITUTIONS.items():
        if special_name in arg_dict:
            # swap it in arg dict
            arg_dict[true_name] = arg_dict[special_name]
            del arg_dict[special_name]

        if special_name in given_shorthands:
            # point the shortcut to the right name
            given_shorthands[true_name] = given_shorthands[special_name]
            del given_shorthands[special_name]

    # Final pass: check for the special args that go to the 'run' command
    # for an experiment grid, separate them from the arg dict, and make sure
    # that they have unique values. The special args are given by RUN_KEYS.
    run_kwargs = dict()
    for k in RUN_KEYS:
        if k in arg_dict:
            val = arg_dict[k]
            assert len(val) == 1, \
                friendly_err("You can only provide one value for %s."%k)
            run_kwargs[k] = val[0]
            del arg_dict[k]

    # Determine experiment name. If not given by user, will be determined
    # by the algorithm name.
    if 'exp_name' in arg_dict:
        assert len(arg_dict['exp_name']) == 1, \
            friendly_err("You can only provide one value for exp_name.")
        exp_name = arg_dict['exp_name'][0]
        del arg_dict['exp_name']
    else:
        exp_name = cmd

    # Make sure that if num_cpu > 1, the algorithm being used is compatible
    # with MPI.
    if 'num_cpu' in run_kwargs and not(run_kwargs['num_cpu'] == 1):
        assert cmd in MPI_COMPATIBLE_ALGOS, \
            friendly_err("This algorithm can't be run with num_cpu > 1.")


    if 'env_name' not in arg_dict:
        arg_dict['env_name'] = ['gym_bm:bbo-v0']

    # Construct and execute the experiment grid.
    eg = ExperimentGrid(name=exp_name)
    for k,v in arg_dict.items():
        eg.add(k, v, shorthand=given_shorthands.get(k))
    eg.run(algo, **run_kwargs)


if __name__ == '__main__':
    """
    This is a wrapper allowing command-line interfaces to individual
    algorithms and the plot / test_policy utilities.

    For utilities, it only checks which thing to run, and calls the
    appropriate file, passing all arguments through.

    For algorithms, it sets up an ExperimentGrid object and uses the
    ExperimentGrid run routine to execute each possible experiment.
    """

    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'
    valid_utils = ['plot', 'animate']
    valid_help = ['--help', '-h', 'help']
    valid_cmds = BASE_ALGO_NAMES + valid_utils + valid_help
    assert cmd in valid_cmds, \
        "Select an algorithm or utility which is implemented in run_rl."

    if cmd in valid_help:
        # Before all else, check to see if any of the flags is 'help'.

        # List commands that are available.
        str_valid_cmds = '\n\t' + '\n\t'.join(BASE_ALGO_NAMES+valid_utils)
        help_msg = dedent("""
            Experiment in RL view of GACEM from the command line with

            \tpython -m gacem.scripts.scripts.run_rl CMD [ARGS...]

            where CMD is a valid command. Current valid commands are:
            """) + str_valid_cmds
        print(help_msg)

        # Provide some useful details for algorithm running.
        subs_list = ['--' + k.ljust(10) + 'for'.ljust(10) + '--' + v \
                     for k,v in SUBSTITUTIONS.items()]
        str_valid_subs = '\n\t' + '\n\t'.join(subs_list)
        special_info = dedent("""
            FYI: When running an algorithm, any keyword argument to the
            algorithm function can be used as a flag, eg

            \tpython -m gacem.scripts.scripts.run_rl ppo --fn rastrigin --ndim 100 --clip_ratio 0.1

            If you need a quick refresher on valid kwargs, get the docstring
            with

            \tpython -m gacem.scripts.scripts.run_rl [algo] --help

            Also: Some common but long flags can be substituted for shorter
            ones. Valid substitutions are:
            """) + str_valid_subs
        print(special_info)

    elif cmd in valid_utils:

        if cmd == 'plot':
            plot_module = ['spinup.run', 'plot']
            args = [sys.executable if sys.executable else 'python', '-m'] + plot_module
        elif cmd is 'animate':
            # Execute the correct utility file.
            runfile = osp.join(osp.abspath(osp.dirname(__file__)), cmd +'.py')
            args = [sys.executable if sys.executable else 'python', runfile]
        else:
            raise ValueError(f'CMD {cmd} is not supported yet!')

        total_args = args + sys.argv[2:]
        subprocess.check_call(total_args, env=os.environ)
    else:
        # Assume that the user plans to execute an algorithm. Run custom
        # parsing on the arguments and build a grid search to execute.
        args = sys.argv[2:]
        parse_and_execute_grid_search(cmd, args)
