"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import gym
import copy
import numpy as np
from utils.misc import sample_continuous_policy
from carnav.env import CarNav

def generate_data(rollouts, data_dir, noise_type, **carnav_kwargs): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    env = CarNav(**carnav_kwargs)
    seq_len = 1000

    for i in range(rollouts):
        env.reset()
        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        done = False
        while not done and t < seq_len:

            action = a_rollout[t]
            t += 1

            s, r, done, _ = env.step(action)
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            if done or t == seq_len - 1:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='white')

    gen_args = ["rollouts", "dir", "policy"]

    parser.add_argument('--width', default= 256, type=int)
    parser.add_argument('--height', default= 256, type=int)
    parser.add_argument('--step-size',default=10, type=int)
    parser.add_argument('--reset-location',default="random", type=str)
    parser.add_argument('--game_id', default=0, type=int)
    parser.add_argument('--pattern', default="-", type=str)

    args = parser.parse_args()
    carnav_kwargs = copy.deepcopy(args)
    for k in gen_args:
        del carnav_kwargs.__dict__[k]
    generate_data(args.rollouts, args.dir, args.policy, **carnav_kwargs.__dict__)
