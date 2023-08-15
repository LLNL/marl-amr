"""Functions for conducting eval episodes without exploration."""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def test(n_eval, alg, env, sess, config):
    """Runs eval episodes.

    Args:
        n_eval: number of episodes to run. This is overridden
            if IC randomization mode is 'discrete'
        alg: one of the alg objects
        env: env object
        sess: TF session
        config: ConfigDict

    Returns: average cumulative reward, error, 
    cumulative_error, num_steps, num_refined
    """
    epsilon = 0
    force = False
    if env.solver.ic_params['randomize'] == 'discrete':
        # override to evaluate every case
        n_eval = len(env.solver.ic_params[env.solver.param_reqs[0]])
        force = True

    if env.multi_objective:
        # global error, dof_budget_balance
        # time_budget_balance, num_steps, num_refined
        # episode return, return_error, return_dof
        measurements = np.zeros((n_eval, 6 + alg.n_objectives))
    else:
        # global error, dof_budget_balance
        # time_budget_balance, num_steps, num_refined
        # episode return
        measurements = np.zeros((n_eval, 6))

    for idx_episode in range(n_eval):

        if env.multi_objective:
            # fixed preference to measure progress across training
            preference = 0.5 * np.ones(alg.n_objectives)

        force_case = idx_episode if force else None
        if alg.name == 'vdgn':
            graph, info = env.reset(force_case)

        done = False
        num_steps = 0
        while not done:
            # ---------------- Run agent(s) -------- #
            if alg.name == 'vdgn':
                if env.multi_objective:
                    actions = alg.run_actor(graph, epsilon, sess, preference)
                else:
                    # mode is only used for noisy net
                    actions = alg.run_actor(graph, epsilon, sess, mode='eval')
                action_dict = {}
                for idx, agent_id in enumerate(
                        info['map_agent_id_to_idx'].keys()):
                    action_dict[agent_id] = actions[idx]
            else:  # test random policy
                actions = alg.run_actor(list_obs)
            # ---------------- Run agent(s) -------- #

            # -------------- Step ----------------#
            if alg.name == 'vdgn':
                graph_next, reward, done, info = env.step(action_dict)
                reward = reward[list(reward.keys())[0]]
                done = done['__all__']
                graph = graph_next
            else:
                list_obs_next, reward, done, info = env.step(actions)
                list_obs = list_obs_next
            # -------------- Step ----------------#

            num_steps += 1
            measurements[idx_episode, 5] += np.sum(reward)
            if env.multi_objective:
                measurements[idx_episode, 6:6+alg.n_objectives] += reward

        measurements[idx_episode, 0] = info['global_error']
        measurements[idx_episode, 1] = info['dof_budget_balance']
        measurements[idx_episode, 2] = info['time_budget_balance']
        measurements[idx_episode, 3] = info['step_count']
        measurements[idx_episode, 4] = info['num_refined_cumulative']
            
    return np.average(measurements, axis=0)


def test_baseline(n_eval, policy, env, error_str='error'):
    """Runs eval episodes.

    Args:
        n_eval: number of episodes to run
        alg: ZZPolicy
        env: env object
        error_str: str key of info dict

    Returns: average cumulative reward, error, 
    cumulative error, num_steps, num_refined
    """
    # episodic return, error, cumulative_error, num_steps, num_refined
    measurements = np.zeros((n_eval, 5))
    epsilon = 0

    for idx_episode in range(1, n_eval + 1):

        obs = env.reset()
        done = False
        num_steps = 0

        while not done:

            action = policy(env.sim)
            obs_next, reward, done, info = env.step(action)

            num_steps += 1
            measurements[idx_episode-1, 0] += reward
            obs = obs_next

        measurements[idx_episode-1, 1] = info[error_str] if error_str in info else 0
        measurements[idx_episode-1, 2] = info['cumulative_error'] if 'cumulative_error' in info else 0
        measurements[idx_episode-1, 3] = num_steps
        measurements[idx_episode-1, 4] = info['num_refined'] if 'num_refined' in info else 0
            
    return np.average(measurements, axis=0)
