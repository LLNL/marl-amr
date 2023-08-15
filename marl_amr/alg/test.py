"""Functions to test trained policies and baseline policies."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time

from marl_amr.alg.utils import main_util
from marl_amr.tools import amr_utils

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def test(config, name_train='', name_test='', idx_seed=0,
         write_csv=True, save_mesh=False, save_mesh_all_steps=False,
         verbose=False, write_pareto=False, save_err_time=False):

    main_util.set_global_seed(config.main.seed)

    dir_name = config.main.dir_name
    dir_restore = config.main.dir_restore
    exp_name = config.main.exp_name
    train_path = config.main.train_path
    log_path = os.path.join(train_path, exp_name, dir_name)
    if write_csv:
        os.makedirs(log_path, exist_ok=True)
    results_path = os.path.join(log_path, '%s_%s_%d.csv' % (
        name_train, name_test, idx_seed))
    restore_path = os.path.join(train_path, dir_restore)
    model_name_restore = config.main.model_name_restore

    if save_err_time:
        path_err_time = os.path.join(dir_restore, 'error_vs_time.csv')

    n_episodes = config.alg.n_test_episodes
    epsilon = 0

    if config.main.exp_name == 'advection':
        from marl_amr.envs.graph_env_h import GraphEnv
        env = GraphEnv(config.env)
        sample_graph, info = env.reset()
        sample_graph_copy = sample_graph
        if env.multi_objective:
            # concat with dummy preference for graphnet initialization
            dummy_w = np.zeros(env.multi_objective)
            new_nodes = np.array([np.concatenate((obs, dummy_w))
                                  for obs in sample_graph[0]])
            sample_graph = (new_nodes, sample_graph[1], sample_graph[2])
    else:
        raise NotImplementedError

    if config.env.multi_objective:
        from marl_amr.alg.vdgn_multiobj import Alg
    else:
        from marl_amr.alg.vdgn import Alg
    alg = Alg(config.alg, config.nn, sample_graph, env.dim_action)

    config_proto = tf.compat.v1.ConfigProto()
    if config.main.use_gpu:
        config_proto.gpu_options.visible_device_list = str(config.main.gpu_id)
        config_proto.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    sess = tf.compat.v1.Session(config=config_proto)
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(restore_path, model_name_restore))

    if write_csv:
        if write_pareto:
            header = 'episode,true_global_error,sum_of_dofs\n'
        else:
            header = ('episode,init_global_error,episode_steps,num_refined,'
                      'sum_of_dofs,global_error,true_global_error,return\n')
        with open(results_path, 'w') as f:
            f.write(header)

    if save_err_time:
        with open(path_err_time, 'w') as f:
            f.write('time,global_error,sum_of_dof\n')

    if env.multi_objective:
        # init_norm, episode_steps, num_refined, sum_of_dofs,
        # global_error, true_global_error, return, return_err, return_dof
        measurements = np.zeros((n_episodes, 9))
    else:
        # init_norm, episode_steps, num_refined, sum_of_dofs,
        # global_error, true_global_error, return
        measurements = np.zeros((n_episodes, 7))
    t_act = 0
    t_env = 0
    n_steps = 0
    for idx_episode in range(1, n_episodes+1):
        # print('Episode', idx_episode)
        if config.alg.name == 'vdgn' and idx_episode == 1:
            # Avoid extra reset so that function initialization
            # of each episode is same as the other methods
            graph = sample_graph_copy
        elif config.alg.name == 'vdgn':
            graph, info = env.reset()

        if env.multi_objective:
            # preference = np.random.randn(env.multi_objective)
            # preference = np.abs(preference) / np.linalg.norm(preference, 1)
            # alpha = np.random.random()
            # preference = np.array([alpha, 1-alpha])
            preference = np.array([0.2, 0.8])
            # preference = np.array([0, 1.0])

        n_step = 0
        if save_mesh_all_steps:
            mesh_dir = '../results/mesh_files'
            amr_utils.output_mesh(env, mesh_dir, n_step)
            
        if save_err_time:
            with open(path_err_time, 'a') as f:
                f.write('{:f},{:e},{:d}\n'.format(
                    0, env.init_global_error[0], env.sum_of_dofs))

        done = False
        episode_return = 0
        if verbose:
            print('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                't', 'dofs', 'reward', 'return', 'error', 'log(error)', 'done'))
        while not done:

            t_start = time.time()
            if config.alg.name == 'vdgn':
                if env.multi_objective:
                    actions = alg.run_actor(graph, epsilon, sess, preference)
                else:
                    # mode is only used for noisy net
                    actions = alg.run_actor(graph, epsilon, sess, mode='eval')
                action_dict = {}
                for idx, agent_id in enumerate(
                        info['map_agent_id_to_idx'].keys()):
                    action_dict[agent_id] = actions[idx]
            else:
                # Run actor network for all agents as batch
                actions = alg.run_actor(list_obs, epsilon, sess)
            t_act += time.time() - t_start
            n_steps += 1
    
            t_env_start = time.time()
            if config.alg.name == 'vdgn':
                graph_next, reward, done, info = env.step(action_dict,
                                                          save_mesh_all_steps)
                # all agents reward are the same, so just take one
                reward = reward[list(reward.keys())[0]]
                done = done['__all__']
                # print('sum of dofs', env.sum_of_dofs)
                # print('reward', reward)
                # input('here')
            else:
                list_obs_next, reward, done, info = env.step(actions)
            t_env += time.time() - t_env_start

            n_step += 1
            episode_return += reward
            if verbose:
                print('%.4f\t%d\t%.2f\t%.2f\t%.4f\t%.4f\t\t%r\n' % (
                    env.solver.t, env.sum_of_dofs, reward, episode_return,
                    info['global_error'], np.log(info['global_error']),
                    done))
            if save_mesh_all_steps:
                amr_utils.output_mesh(env, mesh_dir, str(n_step)+'b')
            if save_err_time:
                with open(path_err_time, 'a') as f:
                    f.write('{:f},{:e},{:d}\n'.format(n_step*env.solver.t_step, info['global_error'],
                                                      env.sum_of_dofs))

            if config.alg.name == 'vdgn':
                graph = graph_next
            else:
                raise NotImplementedError

        if write_csv:
            if write_pareto:
                s = '%d,%.6e,%d\n' % (idx_episode, info['true_global_error'],
                                    info['sum_of_dofs'])
            else:
                s = '%d,%.6e,%d,%d,%d,%.6e,%.6e,%.6e\n' % (
                    idx_episode, info['init_global_error'], n_step,
                    info['num_refined_cumulative'], info['sum_of_dofs'],
                    info['global_error'], info['true_global_error'], episode_return)
            with open(results_path, 'a') as f:
                f.write(s)
        measurements[idx_episode-1, 0] += info['init_global_error']
        measurements[idx_episode-1, 1] += info['step_count']
        measurements[idx_episode-1, 2] += info['num_refined_cumulative']
        measurements[idx_episode-1, 3] += info['sum_of_dofs']
        measurements[idx_episode-1, 4] += info['global_error']
        measurements[idx_episode-1, 5] += info['true_global_error']
        if env.multi_objective:
            measurements[idx_episode-1, 6] += np.sum(np.multiply(
                episode_return, preference))
            measurements[idx_episode-1, 7:9] = episode_return
        else:
            measurements[idx_episode-1, 6] += episode_return

    # print(info['true_global_error'], info['sum_of_dofs'])

    if write_csv and not env.multi_objective:
        measurements = np.average(measurements, axis=0)
        with open(results_path, 'a') as f:
            f.write('avg,%.6e,%.2f,%.2f,%.2f,%.6e,%.6e,%.6e\n' %
                    tuple(measurements))
            f.write('avg_t_action,%.6e\n' % (t_act/n_steps))
            f.write('avg_t_env,%.6e\n' % (t_env/n_steps))
            f.write('avg_t_episode,%.6e\n' % ((t_act + t_env)/n_episodes))
    else:
        print(('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            'init_global_error', 'step_count', 'num_refined_cumulative',
            'sum_of_dofs', 'global_error', 'true_global_error', 'return')).expandtabs(25))
        print(('%.4f\t%d\t%d\t%d\t%.5f\t%.5f\t%.4f\n' % tuple(measurements[-1])).expandtabs(25))

    if save_mesh or save_mesh_all_steps:
        amr_utils.output_mesh(env, mesh_dir, n_step+1)
        
    # print("Average time per action", t_act/(n_episodes*env.max_steps))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_type', type=str,
                        choices=['norefine', 'random', 'tf', 'true_error',
                                 'zz'])
    parser.add_argument('--config_name', type=str, default='advection_test',
                        help='name of config file under configs/')
    parser.add_argument('--write_csv', action='store_true',
                        help='whether to save each episode performance to csv')
    parser.add_argument('--save_mesh', action='store_true',
                        help='whether to save final mesh file')
    parser.add_argument('--save_mesh_all_steps',  action='store_true',
                        help='whether to save mesh at all time steps')
    parser.add_argument('--name_train', type=str, default='',
                        help='name of problem on which policies were trained')
    parser.add_argument('--name_test', type=str, default='',
                        help='name of problem on which policies are tested')
    parser.add_argument('--verbose',  action='store_true',
                        help='whether to print environment information')
    parser.add_argument('--write_pareto',  action='store_true',
                        help='generate Pareto frontier')
    parser.add_argument('--save_err_time',  action='store_true',
                        help='records global error vs solver time')
    args = parser.parse_args()

    if args.write_csv and (args.name_train == '' or args.name_test == ''):
        raise ValueError('Need to specify name_train and name_test '
                         'if writing to csv')

    if args.policy_type in ['norefine', 'random', 'true_error', 'zz']:
        raise NotImplementedError
    elif args.policy_type == 'tf':
        import importlib
        module = importlib.import_module('configs.'+args.config_name)
        config = module.get_config()
        test(config, args.name_train, args.name_test, idx_seed=0,
             write_csv=args.write_csv, save_mesh=args.save_mesh,
             save_mesh_all_steps=args.save_mesh_all_steps,
             verbose=args.verbose, write_pareto=args.write_pareto,
             save_err_time=args.save_err_time)
