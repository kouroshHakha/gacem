from typing import cast

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from pathlib import Path


from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import mpi_fork, proc_id, num_procs, broadcast

import gacem.alg.es.core as core
from gacem.alg.utils.mpi_tools import gather
from gym_bm.envs.bbo_env import BlackBoxOptEnv

from utils.pdb import register_pdb_hook
from utils.hdf5 import save_dict_to_hdf5

register_pdb_hook()


def es(env_fn, *, env_kwargs=dict(),  actor_critic=core.Actor, ac_kwargs=dict(),  seed=0,
       steps_per_epoch=4000, epochs=50, logger_kwargs=dict(), save_freq=10):
    """
    Reinforce algorithm which uses AR models as the policy with no notion of state (Uses Dummy state)
    """

    # Set up logger and save configuration
    output_dir = logger_kwargs['output_dir']
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())


    # Random seed
    seed += 10000 * proc_id()
    np.random.seed(seed)

    # Instantiate environment
    env = cast(BlackBoxOptEnv, env_fn())
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Set up number of trials per resource
    local_steps_per_epoch = int(steps_per_epoch / num_procs())

    # Create actor-critic module
    ac_kwargs.update(num_params=env.dim, popsize=local_steps_per_epoch)
    ac: core.Actor = actor_critic(env.observation_space, env.action_space, **ac_kwargs)


    # Prepare for interaction with environment
    start_time = time.time()
    env.reset(ntimes=local_steps_per_epoch)


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        # run nsamples in parallel or in series depending on the environment step implementation
        # observation here can be anything that gives information about the environment
        # next_obs is totally independent of the current action (like contextual bandit problem)
        acts = ac.ask()
        _, fitnesses, _, info = env.step(acts)

        # create global variables to store the actions and fitnesses to save them in files
        glob_acts = np.zeros((acts.shape[0] * num_procs(), acts.shape[1]))
        glob_fitnesses = np.zeros((fitnesses.shape[0] * num_procs(), ))

        logger.store(EpRet=fitnesses)
        env.reset(ntimes=local_steps_per_epoch)

        gather(acts, glob_acts)
        gather(fitnesses, glob_fitnesses)

        # sanity check to see if gathering worked! comment out if not needed
        idx = slice(proc_id()*local_steps_per_epoch, (proc_id()+1)*local_steps_per_epoch)
        broadcast(glob_acts)
        broadcast(glob_fitnesses)
        assert np.all(glob_acts[idx] == acts), ValueError(f'proc_id {proc_id()} Failed')
        assert np.all(glob_fitnesses[idx] == fitnesses), ValueError(f'proc_id {proc_id()} Failed')

        if proc_id() == 0:
            if (epoch % save_freq * 10 == 0) or (epoch == epochs-1):
                # save actions into h5 file
                path = Path(output_dir) / 'trials' / f'trials{epoch}.h5'
                path.parent.mkdir(parents=True, exist_ok=True)
                data_np = dict(act=glob_acts, fit=glob_fitnesses)
                save_dict_to_hdf5(data_np, path)
                # save state of the env
                logger.save_state({'env': env}, None)

        # Perform ES update!
        ac.tell(fitnesses)
        result = ac.result() # first element is the best solution, second element is the best fitness

        logger.store(Std=ac.rms_stdev(), BestFit=result[1])

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Std', average_only=True)
        logger.log_tabular('BestFit', average_only=True)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, default='rastrigin')
    parser.add_argument('--ndim', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='es')
    parser.add_argument('--ptype', type=str, default='CMAES', choices=core.Actor.VALID_PTYPES)
    args = parser.parse_args()

    if args.ptype == 'CMAES':
        ac_kwargs = dict(weight_decay=0, sigma_init=0.5)
    elif args.ptype == 'OpenES':
        ac_kwargs = dict(sigma_init=0.5,
                         sigma_decay=0.999,
                         learning_rate=0.1,
                         learning_rate_decay = 1.0,
                         antithetic=False,
                         weight_decay=0.00,
                         rank_fitness=False,
                         forget_best=False)
    elif args.ptype == 'SimpleGA':
        ac_kwargs = dict(sigma_init=0.5,        # initial standard deviation
                         elite_ratio=0.1,       # percentage of the elites
                         forget_best=False,     # forget the historical best elites
                         weight_decay=0.00,)     # weight decay coefficient
    elif args.ptype == 'PEPG':
        ac_kwargs = dict(sigma_init=0.5,
                         learning_rate=0.1,
                         learning_rate_decay=1.0,
                         average_baseline=False,
                         weight_decay=0.00,
                         rank_fitness=False,
                         forget_best=False)
    else:
        raise NotImplementedError
    ac_kwargs.update(policy_type=args.ptype)

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    es(env_fn=lambda : gym.make('gym_bm:bbo-v0', dim=args.ndim, fn=args.fn),
       actor_critic=core.Actor,
       ac_kwargs=ac_kwargs,
       seed=args.seed,
       steps_per_epoch=args.steps,
       epochs=args.epochs,
       logger_kwargs=logger_kwargs)