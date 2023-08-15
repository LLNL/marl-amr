"""Defines classes for parallel episode rollouts using Ray."""

import ray

from marl_amr.alg import evaluate
from marl_amr.alg import replay as replay_lib
from marl_amr.envs.graph_env_h import GraphEnv

import numpy as np
import tensorflow as tf

import os
import random
import time


class SequentialRunner:
    """Class that holds env and alg objects."""
    
    def __init__(self, config):
        """Initializes env and alg.

        Args:
            config: ConfigDict
        """
        if config.main.exp_name == 'advection':
            if config.alg.name == 'vdgn':
                self.env = GraphEnv(config.env)
                sample_graph, info = self.env.reset()
                if config.env.multi_objective:
                    # concat with dummy preference for graphnet initialization
                    dummy_w = np.zeros(env.multi_objective)
                    new_nodes = np.array([np.concatenate((obs, dummy_w))
                                          for obs in sample_graph[0]])
                    sample_graph = (new_nodes, sample_graph[1], sample_graph[2])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if config.env.solver.initial_condition.test_is_different:
            from copy import deepcopy
            config_test = deepcopy(config)
            config_test.env.solver.initial_condition.params = (
                config_test.env.solver.initial_condition.params_test)
            self.env_test = GraphEnv(config_test.env)

        if config.alg.name == 'vdgn':
            if config.env.multi_objective:
                from marl_amr.alg.vdgn_multiobj import Alg
            else:
                from marl_amr.alg.vdgn import Alg
            self.alg = Alg(config.alg, config.nn, sample_graph,
                           self.env.dim_action)
        else:
            raise NotImplementedError

        config_proto = tf.compat.v1.ConfigProto()
        if config.main.use_gpu:
            config_proto.gpu_options.visible_device_list = str(config.main.gpu_id)
            config_proto.gpu_options.allow_growth = True
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
        self.sess = tf.compat.v1.Session(config=config_proto)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Initialize target networks to equal main networks
        self.sess.run(self.alg.list_initialize_target_ops)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=config.main.max_to_keep)

        if config.main.resume_training or config.main.resume_from_pretrained:
            prefix = os.path.join(config.main.train_path, config.main.dir_restore)
            path = os.path.join(prefix, config.main.model_name_restore)
            if not os.path.isfile(path + '.index'):
                raise FileNotFoundError('Could not find %s' % path)
            print('Restoring variables from %s' % path)
            self.saver.restore(self.sess, path)

        self.config = config

    def run_episode(self, epsilon, idx_episode, preference=None):
        """Runs one episode.

        Args:
            sess: tf.Session
            env: AMR env object
            alg: algorithm object
            epsilon: float
            idx_episode: int
        """
        t_env = 0
        t_model = 0

        # buffer to store experiences in this episode
        list_transitions = []

        t_env_start = time.time()
        graph, info = self.env.reset()
        t_env += time.time() - t_env_start

        done = False
        step = 0
        while not done:

            t_model_start = time.time()
            if self.config.env.multi_objective:
                actions = alg.run_actor(graph, epsilon, self.sess, preference)
            else:
                actions = self.alg.run_actor(graph, epsilon, self.sess)
            action_dict = {}
            for idx, agent_id in enumerate(
                    info['map_agent_id_to_idx'].keys()):
                action_dict[agent_id] = actions[idx]
            t_temp = time.time()
            t_model += t_temp - t_model_start

            graph_next, reward, done, info = self.env.step(action_dict)
            # all agents reward are the same, so just take one
            reward = reward[list(reward.keys())[0]]
            done = done['__all__']
            step += 1
            t_env += time.time() - t_temp

            if self.config.alg.prioritized_replay:
                transition = replay_lib.TransitionGraph(
                    nodes_t=graph[0], edges_t=graph[1], adj_t=graph[2],
                    n_t=len(graph[0]), a_t=actions, r_t=reward,
                    nodes_tp1=graph_next[0], edges_tp1=graph_next[1],
                    adj_tp1=graph_next[2], n_tp1=len(graph_next[0]),
                    done=done)
            else:
                # node array, edge array, adj_matrix, n_agents, actions
                # reward, next node array, next edge array, next adj_matrix
                # n_agents next, done
                transition = np.array([graph[0], graph[1], graph[2],
                                       len(graph[0]), actions, reward,
                                       graph_next[0], graph_next[1],
                                       graph_next[2], len(graph_next[0]),
                                       done], dtype=object)

            list_transitions.append(transition)

            graph = graph_next

        return list_transitions, t_env, t_model, step

    def train(self, batch, weights=None):
        """Gradient update using replay buffer.

        Args:
            replay: PrioritizedTransitionReplay

        Returns:
            td_errors
        """
        td_errors = self.alg.train(batch, self.sess, weights)

        return td_errors
    
    def evaluate(self):
        """Runs evaluation episodes.

        Returns:
            (global_error, dof_budget_balance,
             time_budget_balance, n_steps, num_refined,
             r_eval)
        """
        if self.config.env.solver.initial_condition.test_is_different:
            env = self.env_test
        else:
            env = self.env
        return evaluate.test(self.config.alg.n_eval, self.alg, env,
                             self.sess, self.config)

    def update_beta(self):
        self.alg.update_beta()

    def save(self, save_path=None):

        self.saver.save(self.sess, save_path)

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)
        

# @ray.remote(num_gpus=0.1) # uncomment to train on GPUs
@ray.remote
class ParallelRunner(SequentialRunner):

    def __init__(self, config):

        super().__init__(config)

    def get_weights(self):
        """Gets neural net params."""

        q_i, q_i_target = self.sess.run([self.alg.q_i_var,
                                         self.alg.q_i_target_var])

        weights = {
            'q_i': q_i,
            'q_i_target': q_i_target
        }        

        return weights

    def set_weights(self, weights):
        """Replaces neural network weights with the given weights.

        Args:
            weights: map with keys 'q_i' and 'q_i_target'
        """

        map_param_to_ph_and_op = {
            'q_i': (self.alg.list_q_i_ph,
                    self.alg.list_set_q_ops),
            'q_i_target': (self.alg.list_q_i_target_ph,
                           self.alg.list_set_q_target_ops)
        }

        feed = {}
        set_weight_ops = []

        for key, weight in weights.items():
            # weight is a group of primitive vars, e.g. layer weights and biases
            ph, op = map_param_to_ph_and_op[key]
            set_weight_ops.append(op)
            for i in range(len(ph)):
                feed[ph[i]] = weight[i]

        self.sess.run(set_weight_ops, feed_dict=feed)

    def evaluate(self, force_case=None):
        """

        Returns:
        global_error, dof_budget_balance, time_budget_balance, step_count,
        num_refined_cumulative, return
        """
        if self.config.env.solver.initial_condition.test_is_different:
            env = self.env_test
        else:
            env = self.env

        epsilon = 0
        if env.multi_objective:
            measurements = np.zeros(6 + self.alg.n_objectives)
        else:
            measurements = np.zeros(6)

        if env.multi_objective:
            # fixed preference to measure progress across training
            preference = 0.5 * np.ones(self.alg.n_objectives)

        graph, info = env.reset(force_case)

        done = False
        num_steps = 0
        while not done:
            # ---------------- Run agent(s) -------- #
            if env.multi_objective:
                actions = self.alg.run_actor(graph, epsilon, self.sess, preference)
            else:
                # mode is only used for noisy net
                actions = self.alg.run_actor(graph, epsilon, self.sess, mode='eval')
            action_dict = {}
            for idx, agent_id in enumerate(
                    info['map_agent_id_to_idx'].keys()):
                action_dict[agent_id] = actions[idx]
            # ---------------- Run agent(s) -------- #

            # -------------- Step ----------------#
            graph_next, reward, done, info = env.step(action_dict)
            reward = reward[list(reward.keys())[0]]
            done = done['__all__']
            graph = graph_next
            # -------------- Step ----------------#

            num_steps += 1
            measurements[5] += np.sum(reward)
            if env.multi_objective:
                measurements[6:6+self.alg.n_objectives] += reward

        measurements[0] = info['global_error']
        measurements[1] = info['dof_budget_balance']
        measurements[2] = info['time_budget_balance']
        measurements[3] = info['step_count']
        measurements[4] = info['num_refined_cumulative']

        return measurements


class Runner:
    """Manages runners, syncs weights."""

    def __init__(self, config):
        """Initializes runners.

        Args:
            config: ConfigDict
        """
        self.config = config
        self.n_parallel = config.main.n_parallel

        if self.n_parallel == 1:
            self.runners = [SequentialRunner(config)]
        else:
            self.runners = [ParallelRunner.remote(config)
                            for i in range(self.n_parallel)]

        # for parallel eval episodes
        if config.env.solver.initial_condition.params.randomize == 'discrete':
            param_map = self.config.env.solver.initial_condition.params
            # value = param_map[list(param_map.keys())[0]] # use any key-value pair
            self.n_cases_eval = len(param_map['x0_discrete'])
        elif config.env.solver.initial_condition.params.randomize == 'uniform':
            self.n_cases_eval = config.alg.n_eval

    def get_preferences(self):
        """Samples preference weights."""
        alpha = np.random.random(self.n_parallel)
        preferences = np.transpose(
            np.vstack([alpha, 1-alpha]), [1,0])
        return preferences

    def run_episode(self, epsilon, idx_episode):
        """Runs single or parallel envs for complete episodes.

        Args:
            epsilon: float
            idx_episode: int

        Returns:
            list of parallel envs, each entry is a tuple of the form
            (list_transitions, t_env, t_model, steps)
        """
        if self.n_parallel == 1:
            if self.config.env.multi_objective:
                preferences = self.get_preferences()
                return [self.runners[0].run_episode(epsilon, idx_episode,
                                                    preferences[0])]
            else:
                return [self.runners[0].run_episode(epsilon, idx_episode)]
        else:
            # List over parallel episodes of tuples
            # (list_transitions, t_env, t_model, step)
            if self.config.env.multi_objective:
                return ray.get([r.run_episode.remote(epsilon, idx_episode, p)
                                for r, p in zip(self.runners, preferences)])
            else:
                return ray.get([r.run_episode.remote(epsilon, idx_episode)
                                for r in self.runners])

    def train(self, batch, priority_weights=None):
        """Gradient update using replay buffer.

        If n_parallel > 1, then take one runner, update its model, then
        set model of all other runners equal to it.

        Args:
            batch: batch sampled from PrioritizedTransitionReplay
            priority_weights: weights from PrioritizedTransitionReplay

        Returns:
            td_errors: np.array of values for each transition in batch (
                sampled from replay)
        """
        if self.n_parallel == 1:
            return self.runners[0].train(batch, priority_weights)
        else:
            td_errors = ray.get(
                self.runners[0].train.remote(batch, priority_weights))

        # Get the updated weights of runner 0
        weights = ray.get(self.runners[0].get_weights.remote())
        # Set other runners equal to those weights
        futures = [r.set_weights.remote(weights) for r in self.runners[1:]]
        ray.get(futures)

        return td_errors

    def train_parallel(self, list_batch, list_priority_weights=None):
        """Parallel gradient updates.

        Args:
            list_batch: list of batches, each sampled from
                PrioritizedTransitionReplay
            list_priority_weights: list of weights, each sampled from
                PrioritizedTransitionReplay

        Returns:
            list_td_errors: list of np.array of values for each transition
        """
        # Compute weight updates in parallel
        n_parallel = len(list_batch)
        list_td_errors = ray.get([r.train.remote(batch, priority_weights) for
                                  r, batch, priority_weights in zip(
                                      self.runners[:n_parallel], list_batch,
                                      list_priority_weights)])
        # Get all updated weights
        # list of maps with keys 'q_i', 'q_i_target'
        # value for 'q_i' is a list of individual nn variables
        weights = ray.get([r.get_weights.remote()
                           for r in self.runners[:n_parallel]])
        # Average the weights, then set all runners equal to those weights
        avg_weights = {}
        for key in ['q_i', 'q_i_target']:
            n_vars = len(weights[0][key])  # get an example list of vars
            list_vars_avg = [0] * n_vars
            for idx_var in range(n_vars): # go through each nn variable
                # for this var, get list of updated versions across all runs
                updated_vars = [weights[idx_run][key][idx_var] for idx_run in
                                range(n_parallel)]
                list_vars_avg[idx_var] = np.average(updated_vars, axis=0)
            avg_weights[key] = list_vars_avg

        # Set all runners equal to averaged weights
        ray.get([r.set_weights.remote(avg_weights) for r in self.runners])

        return list_td_errors

    def update_beta(self):
        if self.n_parallel == 1:
            self.runners[0].update_beta()
        else:
            ray.get([r.update_beta() for r in self.runners])

    def save(self, save_path=None):
        if self.n_parallel == 1:
            self.runners[0].save(save_path)
        else:
            ray.get(self.runners[0].save.remote(save_path))

    def load(self, load_path):
        if self.n_parallel == 1:
            self.runners[0].load(load_path)
        else:
            ray.get([r.load.remote(load_path) for r in runners])

    def evaluate(self):

        if self.config.env.solver.initial_condition.test_is_different:
            randomize_type = self.config.env.solver.initial_condition.params_test.randomize
        else:
            randomize_type = self.config.env.solver.initial_condition.params.randomize

        if self.n_parallel == 1:
            return self.runners[0].evaluate()
        else:
            if randomize_type == 'discrete':
                # run all cases in parallel
                list_measurements = ray.get(
                    [r.evaluate.remote(idx_case) for idx_case, r
                     in enumerate(self.runners[: self.n_cases_eval])])
                return np.average(list_measurements, axis=0)
            else:
                list_measurements = ray.get(
                    [r.evaluate.remote() for idx_case, r
                     in enumerate(self.runners[: self.n_cases_eval])])
                return np.average(list_measurements, axis=0)
