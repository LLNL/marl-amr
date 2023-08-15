"""Entry point for off-policy training."""

import numpy as np
import tensorflow as tf

import csv
import glob
import json
import os
import random
import shutil
import time

from marl_amr.alg import evaluate
from marl_amr.alg import replay as replay_lib
from marl_amr.alg import runners
from marl_amr.alg.utils import main_util
from marl_amr.alg.utils import parts
from marl_amr.alg.utils import replay_buffer


def train_function(config):

    main_util.set_global_seed(config.main.seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    train_path = config.main.train_path
    log_path = os.path.join(train_path, exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    if not config.main.resume_training:
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = config.alg.n_episodes
    n_eval = config.alg.n_eval
    period = config.alg.period
    
    epsilon_step = ((config.alg.epsilon_start - config.alg.epsilon_end) /
                    config.alg.epsilon_div)
    if not config.main.resume_training:
        epsilon = config.alg.epsilon_start
    else:
        # Amount that epsilon has decreased during previous training
        epsilon_decrease = config.main.resume_episode * epsilon_step
        epsilon = max(config.alg.epsilon_end,
                      config.alg.epsilon_start - epsilon_decrease)

    runner = runners.Runner(config)

    # Write log headers
    if not config.main.resume_training:
        header = ('episode,step_train,step,time,t_env,t_model,'
                  'episode_steps,num_refined,dof_budget_balance,'
                  'time_budget_balance,r_eval,global_error')
        if config.env.multi_objective:
            header += ',r_err,r_dof\n'
        else:
            header += '\n'
        with open(os.path.join(log_path, 'log.csv'), 'w') as f:
            f.write(header)
    else:
        log_path = os.path.join(train_path, config.main.dir_restore)
        cleaned_log = []
        with open(os.path.join(log_path, 'log.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            cleaned_log.append(next(reader))
            for r in reader:
                if int(r[0]) <= config.main.resume_episode:
                    cleaned_log.append(r)
                    t_start = time.time() - float(r[3])
                    t_env_prev = float(r[4])
                    t_model_prev = float(r[5])
                    step_prev = int(r[2])
        with open(os.path.join(log_path, 'log.csv'), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(cleaned_log)

    if config.alg.prioritized_replay:
        max_seen_priority = 1.0
        importance_sampling_exponent_schedule = parts.LinearSchedule(
            begin_t=config.alg.batch_size,
            end_t=config.alg.n_episodes * (config.env.t_final /
                                           config.env.solver.t_step),
            begin_value = config.alg.priority_importance_exponent_start,
            end_value=1.0)
        replay_structure = replay_lib.TransitionGraph(
            nodes_t=None, edges_t=None, adj_t=None,
            n_t=None, a_t=None, r_t=None,
            nodes_tp1=None, edges_tp1=None, adj_tp1=None,
            n_tp1=None, done=None)
        replay = replay_lib.PrioritizedTransitionReplay(
            config.alg.buffer_size, replay_structure,
            config.alg.priority_exponent,
            importance_sampling_exponent_schedule,
            config.alg.uniform_sample_probability,
            config.alg.priority_normalize_weights,
            np.random.RandomState(config.main.seed), None, None)
    buf = replay_buffer.ReplayBufferNumpy(size=config.alg.buffer_size)
    if config.main.resume_training:
        path = glob.glob(os.path.join(log_path, 'buf_*.npy'))[0] # exists only 1
        with open(path, 'rb') as f:
            mem = np.load(f, allow_pickle=True)
        buf.memory = mem
        filename = path.split('/')[-1] # buf_episode_{}_filled_{}_idx_{}.npy
        buf.idx = int(filename.split('.')[0].split('_')[-1])
        buf.num_filled = int(filename.split('.')[0].split('_')[-3])

    if config.alg.multi_step:
        from marl_amr.alg import multi_step_buffer
        buf_multi_step = multi_step_buffer.NStepReturns(
            config.alg.multi_step, config.alg.gamma)
    
    best_return = -np.inf
    best_model_name = ''
    if not config.main.resume_training:
        idx_episode = 0
        t_start = time.time()
        t_env = 0
        t_model = 0
        step = 0
        step_train = 0
    else:
        idx_episode = config.main.resume_episode
        t_env = t_env_prev
        t_model = t_model_prev
        step = step_prev
        step_train = idx_episode
    env_steps_since_last_train = 0
    # Evaluate the initial policy before training
    (best_return, best_model_name, saved_model) = evaluate_and_save(
        idx_episode, config, runner,
        step_train, step, t_start, t_env, t_model,
        best_return, best_model_name, log_path, False)
    while idx_episode < n_episodes:
        # print('Episode', idx_episode)
        saved_model = False

        # List over parallel episodes of tuples
        # (list_transitions, t_env, t_model, step)
        list_episodes = runner.run_episode(epsilon, idx_episode)

        idx_episode += config.main.n_parallel
        total_env_steps = np.sum([tup[3] for tup in list_episodes])
        step += total_env_steps
        env_steps_since_last_train += total_env_steps
        # get the episode index with longest total time among all parallel runs
        list_total_time = [tup[1] + tup[2] for tup in list_episodes]
        idx_max = np.argmax(list_total_time)
        t_env += list_episodes[idx_max][1]
        t_model += list_episodes[idx_max][2]

        # multi-step returns not yet implemented, requires passing buffer into
        # the remote workers
        for tup in list_episodes:
            list_transitions = tup[0]
            if config.alg.prioritized_replay:
                for trans in list_transitions:
                    replay.add(trans, priority=max_seen_priority)
            else:
                for trans in list_transitions:
                    buf.add(trans)

        n_in_buffer = (replay.size if config.alg.prioritized_replay
                       else len(buf))

        if (n_in_buffer >= config.alg.batch_size and
            env_steps_since_last_train >= config.alg.steps_per_train):
            # Since parallel version runs complete episodes,
            # need to calculate the equivalent number of gradient steps
            # that would have been taken if ran sequentially
            num_train_steps_in_sequential = (env_steps_since_last_train //
                                             config.alg.steps_per_train)
            # carry over the remainder to the next train step
            env_steps_since_last_train = (env_steps_since_last_train -
                                          num_train_steps_in_sequential *
                                          config.alg.steps_per_train)

            t_model_start = time.time()
            if config.main.n_train_parallel:
                idx_start = 0
                # print('num sequential', num_train_steps_in_sequential)
                while idx_start < num_train_steps_in_sequential:
                    idx_end = min(idx_start + config.main.n_train_parallel,
                                  num_train_steps_in_sequential)
                    # print('idx_start', idx_start)
                    # print('idx_end', idx_end)
                    list_batch = []
                    list_indices = []
                    list_priority_weights = []
                    for _ in range(idx_end - idx_start):
                        batch, indices, weights = replay.sample(config.alg.batch_size)
                        list_batch.append(batch)
                        list_indices.append(indices)
                        list_priority_weights.append(weights)
                    list_td_errors = runner.train_parallel(list_batch, list_priority_weights)
                    list_priorities = [np.abs(td_errors) for td_errors in list_td_errors]
                    max_seen_priority = np.max(np.hstack(list_priorities + [max_seen_priority]))
                    for indices, priorities in zip(list_indices, list_priorities):
                        replay.update_priorities(indices, priorities)
                    idx_start = idx_end
                step_train += num_train_steps_in_sequential
            else:
                for _ in range(num_train_steps_in_sequential):
                    if config.alg.prioritized_replay:
                        batch, indices, weights = replay.sample(
                            config.alg.batch_size)
                        td_errors = runner.train(batch, weights)
                        priorities = np.abs(td_errors)
                        max_priority = np.max(priorities)
                        max_seen_priority = np.max([max_seen_priority, max_priority])
                        replay.update_priorities(indices, priorities)
                    else:
                        batch = buf.sample_batch(config.alg.batch_size)
                        runner.train(batch, sess)
                    step_train += 1
            t_model += time.time() - t_model_start

        if config.env.multi_objective:
            runner.update_beta()

        if (n_in_buffer >= config.alg.batch_size and
            epsilon > config.alg.epsilon_end):
            epsilon -= config.main.n_parallel * epsilon_step
    
        if idx_episode % period == 0:
            (best_return, best_model_name, saved_model) = evaluate_and_save(
                idx_episode, config, runner,
                step_train, step, t_start, t_env, t_model,
                best_return, best_model_name, log_path,
                saved_model)

        if idx_episode % save_period == 0:
            if not saved_model:
                runner.save(os.path.join(log_path, '%s.%d' % (
                    model_name, idx_episode)))
                saved_model = True
            f = glob.glob(os.path.join(log_path, 'buf_*.npy'))
            if len(f):
                os.remove(f[0]) # there should only be one such file
            with open(os.path.join(
                    log_path, 'buf_episode_{}_filled_{}_idx_{}.npy'.format(
                        idx_episode, buf.num_filled, buf.idx)), 'wb') as f:
                np.save(f, buf.memory)
    
    if not saved_model:
        runner.save(os.path.join(log_path, model_name))


def evaluate_and_save(idx_episode, config, runner,
                      step_train, step, t_start, t_env, t_model,
                      best_return, best_model_name, log_path,
                      saved_model):
    # Evaluation
    # print("Evaluating\n")
    if config.env.multi_objective:
        (global_error, dof_budget_balance, time_budget_balance, n_steps,
         num_refined, r_eval, r_err, r_dof) = evaluate.test(
             config.alg.n_eval, runner.alg, runner.env, runner.sess, config)
    else:
        (global_error, dof_budget_balance,
         time_budget_balance, n_steps, num_refined,
         r_eval) = runner.evaluate()

    s = '%d,%d,%d,%d,%d,%d,%d,%d,%.3e,%.3e,%.4e,%.4e' % (
        idx_episode, step_train, step,
        time.time()-t_start, t_env, t_model,
        n_steps, num_refined, dof_budget_balance,
        time_budget_balance, r_eval, global_error)
    if config.env.multi_objective:
        s += ',%.4e,%.4e\n' % (r_err, r_dof)
    else:
        s += '\n'

    with open(os.path.join(log_path, 'log.csv'), 'a') as f:
        f.write(s)

    # if global_error <= config.main.save_threshold:
    # if r_eval >= config.main.save_threshold:
    if r_eval >= best_return:
        # saves .data, .index, .meta files
        runner.save(os.path.join(log_path, 'model_best_%d'% idx_episode))
        best_return = r_eval
        # delete previous best
        if best_model_name != '':
            for path in glob.glob(os.path.join(log_path, best_model_name+'*')):
                os.remove(path)
        # copy files to avoid deletion due to TF's max_to_keep
        for path in glob.glob(os.path.join(
                log_path, 'model_best_%d*'%idx_episode)):
            dest = path.replace('model_best', 'mb')
            shutil.copyfile(path, dest)
        best_model_name = 'mb_%d'%idx_episode
        saved_model = True

    return best_return, best_model_name, saved_model


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        default='advection_vdgn',
                        help='name of config file under configs/')
    parser.add_argument('--dir_name', type=str)
    parser.add_argument('--dof_threshold', type=int)
    parser.add_argument('--epsilon_div', type=int)
    parser.add_argument('--epsilon_end', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n_train_parallel', type=int)
    parser.add_argument('--n_heads', type=int)
    parser.add_argument('--num_att_layers', type=int)
    parser.add_argument('--num_recurrent_passes', type=int)
    parser.add_argument('--output_independent', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    import importlib
    module = importlib.import_module('configs.'+args.config_name)
    config = module.get_config()

    cmd = 'nvidia-smi'
    import subprocess as sp
    try:
        output = sp.check_output(cmd.split())
    except:
        config.main.use_gpu = False

    if args.dir_name:
        config.main.dir_name = args.dir_name
    if args.dof_threshold:
        config.env.dof_threshold = args.dof_threshold
    if args.epsilon_div:
        config.alg.epsilon_div = args.epsilon_div
    if args.epsilon_end:
        config.alg.epsilon_end = args.epsilon_end
    if args.lr:
        config.alg.lr = args.lr
    if args.n_train_parallel:
        config.main.n_train_parallel = args.n_train_parallel
    if args.n_heads:
        config.nn.n_heads = args.n_heads
    if args.num_att_layers:
        config.nn.num_att_layers = args.num_att_layers
    if args.num_recurrent_passes:
        config.nn.num_recurrent_passes = args.num_recurrent_passes
    if args.output_independent != None:
        config.nn.output_independent = bool(args.output_independent)
    if args.seed:
        config.main.seed = args.seed

    train_function(config)
