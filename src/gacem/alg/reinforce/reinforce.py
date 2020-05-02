import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from pathlib import Path


from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import gacem.alg.reinforce.core as core

import pdb

from utils.pdb import register_pdb_hook
from utils.hdf5 import save_dict_to_hdf5

register_pdb_hook()


class Buffer:
    """
    A buffer for storing samples and their values experienced by a Reinforce agent interacting
    with the environment, and fitness shaping for calculating the advantages of each action.
    """

    def __init__(self, obs_dim, act_dim, size, lperc):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.lperc = lperc

    def store(self, obs, act, rew, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self):
        """
        Call this at the end of sampling. This looks back in the buffer to where the
        sampling started, and uses rewards from the entire sampling process to compute
        advantage estimates.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = self.rew_buf[path_slice]

        self.adv_buf[path_slice] = core.advantages(rews, lperc=self.lperc)

        self.path_start_idx = self.ptr

    def get(self, return_np=False):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

        data_torch = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        if return_np:
            return data_torch, data
        return data_torch


def reinforce(env_fn, *, env_kwargs=dict(),  actor_critic=core.Actor, ac_kwargs=dict(),  seed=0,
              steps_per_epoch=4000, epochs=50, lperc=0.1, pi_lr=3e-4,
              logger_kwargs=dict(), save_freq=10, linear_aneal=False):
    """
    Reinforce algorithm which uses AR models as the policy with no notion of state (Uses Dummy state)
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    print(logger.output_dir)

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = core.count_vars(ac.pi)
    logger.log(f'\nNumber of parameters: \t pi: {var_counts} \n')

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = Buffer(obs_dim, act_dim, local_steps_per_epoch, lperc)

    # Set up function for computing policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        logp = ac.pi.get_logp(act)
        loss_pi = -(logp * adv.to(logp)).mean()

        # Useful extra info
        approx_kl = (logp_old.to(logp) - logp).mean().item()
        ent = (-logp).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.parameters(), lr=pi_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # setup sigma scheduling
    sigma_init = ac_kwargs.get('fixed_sigma', None)

    def update():
        ac.train()
        data, data_np = buf.get(return_np=True)

        # # store data points along the way
        # if (epoch % save_freq == 0) or (epoch == epochs-1):
        #     path = Path(logger.output_dir) / 'trials' / f'trials{epoch}.h5'
        #     path.parent.mkdir(parents=True, exist_ok=True)
        #     save_dict_to_hdf5(data_np, path)

        # Get loss and info values before update
        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()
        pi_l_old = pi_l_old.item()

        loss_pi, pi_info = compute_loss_pi(data)

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    obss = env.reset(ntimes=local_steps_per_epoch)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        # update sigma according to a linear schedule
        if sigma_init and linear_aneal:
            sigma = ac.pi.fixed_sigma = max(sigma_init - sigma_init * epoch / epochs , 1e-2)
            logger.store(Sigma=sigma)
        # run nsamples in parallel or in series depending on the environement step implementation
        # observation here can be anything that gives information about the environement
        # next_obs is totally independent of the current action (like contextual bandit problem)
        ac.eval()
        acts, logps = ac.act(torch.as_tensor(obss, dtype=torch.float32))

        acts = acts.cpu().numpy()
        logps = logps.cpu().numpy()
        _, rews, _, info = env.step(acts)


        for i in range(local_steps_per_epoch):
            # save and log
            buf.store(obss[i], acts[i], rews[i], logps[i])


        logger.store(EpRet=rews)
        buf.finish_path()
        obss = env.reset(ntimes=local_steps_per_epoch)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform Reinforce update!
        update()


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        if 'Sigma' in logger.log_headers:
            logger.log_tabular('Sigma', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, default='rastrigin')
    parser.add_argument('--ndim', type=int, default=2)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lperc', type=float, default=0.1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='reinforce')
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--aneal', action='store_true', default=False)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    reinforce(env_fn=lambda : gym.make('gym_bm:bbo-v0', dim=args.ndim, fn=args.fn),
              actor_critic=core.Actor,
              ac_kwargs=dict(hidden_sizes=[args.hid]*args.l,
                             bsize=16,
                             fixed_sigma=args.sigma,
                             ),
              linear_aneal=args.aneal,
              lperc=args.lperc,
              seed=args.seed,
              pi_lr=args.lr,
              steps_per_epoch=args.steps,
              epochs=args.epochs,
              logger_kwargs=logger_kwargs)