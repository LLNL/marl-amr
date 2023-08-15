"""Value decomposition graph network."""

from marl_amr.alg import networks
from marl_amr.alg import vdn
from marl_amr.alg.utils import marl_util

from graph_nets import utils_np
from graph_nets import utils_tf
import numpy as np
import tensorflow as tf


class Alg(vdn.Alg):

    def __init__(self, config, config_nn, graph, dim_action):
        """
        Args:
            config: ConfigDict object with algorithm hyperparameters
            config_nn: ConfigDict object for neural network
            graph: tuple of (np.array, np.array, np.array)
                   interpreted as (node attributes, edge attributes, adjacency matrix)
            dim_action: int
        """
        self.sample_data_dict = {'globals': 0, # unused
                                 'nodes': graph[0],
                                 'edges': graph[1],
                                 'senders': np.nonzero(graph[2])[0],
                                 'receivers': np.nonzero(graph[2])[1]}

        super().__init__(config, config_nn, None, dim_action)

    def create_networks(self):

        # Create a placeholder using a graph as template.
        self.graphs_tuple_ph = utils_tf.placeholders_from_data_dicts(
            [self.sample_data_dict])

        self.ragged_row_lengths = tf.compat.v1.placeholder(
            tf.int32, [None], 'ragged_row_lengths')

        self.actions_1hot = tf.compat.v1.placeholder(tf.float32,
                                           [None, self.dim_action],
                                           'actions_1hot')

        with tf.compat.v1.variable_scope('q_i_main'):
            # [batch*(nodes), dim_action]
            if self.config_nn.attention:
                self.q_i = networks.graph_attention_q(
                    self.graphs_tuple_ph, self.config_nn, self.dim_action,
                    self.dueling)
            else:
                self.q_i, self.list_nl = networks.graph_local_q(
                    self.graphs_tuple_ph, self.config_nn, self.dim_action,
                    self.dueling, self.noisy_net, 'train')
        with tf.compat.v1.variable_scope('q_i_target'):
            if self.config_nn.attention:
                self.q_i_target = networks.graph_attention_q(
                    self.graphs_tuple_ph, self.config_nn, self.dim_action,
                    self.dueling)
            else:
                self.q_i_target, self.list_nl_target = networks.graph_local_q(
                    self.graphs_tuple_ph, self.config_nn, self.dim_action,
                    self.dueling, self.noisy_net, 'train')

        self.argmax_q = tf.argmax(self.q_i, axis=1)

        if self.noisy_net:
            # same weights as self.q_i, except with noise turned off
            # this is used for evaluation
            with tf.compat.v1.variable_scope('q_i_main', reuse=True):
                self.q_i_eval, _ = networks.graph_local_q(
                    self.graphs_tuple_ph, self.config_nn, self.dim_action,
                    self.dueling, self.noisy_net, 'eval')
            self.argmax_q_eval = tf.argmax(self.q_i_eval, axis=1)

        # [batch*(nodes)]
        self.q_i_selected = tf.reduce_sum(tf.multiply(
            self.q_i, self.actions_1hot), axis=1)
        q_i_ragged = tf.RaggedTensor.from_row_lengths(
            self.q_i_selected, row_lengths=self.ragged_row_lengths)
        # Without this, an error appears in create_train_op()
        q_i_ragged = q_i_ragged.to_tensor(default_value = 0)
        # For each graph, sum selected q-value over all nodes
        self.q_global = tf.reshape(tf.reduce_sum(q_i_ragged, axis=1), [-1])

        q_i_target_selected = tf.reduce_sum(tf.multiply(
            self.q_i_target, self.actions_1hot), axis=1)
        q_i_target_selected_ragged = tf.RaggedTensor.from_row_lengths(
            q_i_target_selected, row_lengths=self.ragged_row_lengths)
        self.q_global_target = tf.reduce_sum(q_i_target_selected_ragged, axis=1)

        self.q_i_target_max = tf.math.reduce_max(
            self.q_i_target, axis=1)
        q_i_target_ragged = tf.RaggedTensor.from_row_lengths(
            self.q_i_target_max, row_lengths=self.ragged_row_lengths)
        self.q_global_target_max = tf.reduce_sum(q_i_target_ragged, axis=1)

    def reset_noise(self):
        """Sample a new noise for noisy layers."""
        for layer in self.list_nl:
            layer.reset_noise(tf.float32)

    def run_actor(self, graph, epsilon, sess, mode='train'):
        """Gets actions for all agents as a batch.
        
        Args:
            graph: tuple of (nodes, edges, adj_matrix)
            epsilon: float, epsilon-greedy exploration
            sess: TF session
            mode: str, 'train' or 'eval', only used for noisy net

        Returns: np.array of discrete actions
        """
        n_agents = len(graph[0])
        data_dict = {'globals': 0, # unused
                     'nodes': graph[0].astype(np.float32),  # np.array of node attributes
                     'edges': graph[1].astype(np.float32),  # np.array of edge attributes
                     # indices of sender nodes for each edge
                     'senders': np.nonzero(graph[2])[0].astype(np.int32),
                     # indices of receiver nodes for each edge
                     'receivers': np.nonzero(graph[2])[1].astype(np.int32)}
        graph_tuple = utils_np.data_dicts_to_graphs_tuple([data_dict])
        feed_dict = utils_tf.get_feed_dict(
            self.graphs_tuple_ph, utils_np.data_dicts_to_graphs_tuple([data_dict]))

        if self.noisy_net and mode == 'train':
            return sess.run(self.argmax_q, feed_dict=feed_dict)
        elif self.noisy_net and mode == 'eval':
            return sess.run(self.argmax_q_eval, feed_dict=feed_dict)
        elif mode == 'eval':
            return sess.run(self.argmax_q, feed_dict=feed_dict)

        if self.explore_type == 'global':
            if np.random.rand(1) < epsilon:
                return np.random.randint(0, self.dim_action, n_agents)
            else:
                return sess.run(self.argmax_q, feed_dict=feed_dict)
        elif self.explore_type == 'independent':
            actions_argmax = sess.run(self.argmax_q, feed_dict=feed_dict)
            actions = np.zeros(n_agents, dtype=int)
            for idx in range(n_agents):
                if np.random.rand(1) < epsilon:
                    actions[idx] = np.random.randint(0, self.dim_action)
                else:
                    actions[idx] = actions_argmax[idx]

        return actions

    def train(self, batch, sess, weights=None):
        """One training step.

        Args: 
            batch: Case 1: np.array of transitions, each is an np.array of
                   (0. node array, 1. edge array, 2. adj_matrix,
                    3. n_agents, 4. actions, 5. reward,
                    6. next node array, 7. next edge array, 8. next adj_matrix,
                    9. n_agents next, 10. done)
                   Case 2: If using prioritized replay, then batch is a
                       list of TransitionGraph with attributes
                       nodes_t, edges_t, adj_t, n_t, a_t, r_t,
                       nodes_tp1, edges_tp1, adj_tp1, n_tp1, done
            sess: TF session
            weights: 1D np.array of priority weights of samples in batch

        Returns:
            np.array of TD errors, if using prioritized replay, else None
        """
        list_data_dict = []
        list_data_dict_next = []
        if self.prioritized:
            list_actions = []
            (n_agents, reward, n_agents_next,
             done) = [np.empty(self.batch_size) for _ in range(4)]
            for idx, t in enumerate(batch):
                data_dict = {'globals': 0,
                             'nodes': t.nodes_t,
                             'edges': t.edges_t,
                             'senders': np.nonzero(t.adj_t)[0],
                             'receivers': np.nonzero(t.adj_t)[1]}
                list_data_dict.append(data_dict)
                data_dict_next = {'globals': 0,
                                  'nodes': t.nodes_tp1,
                                  'edges': t.edges_tp1,
                                  'senders': np.nonzero(t.adj_tp1)[0],
                                  'receivers': np.nonzero(t.adj_tp1)[1]}
                list_data_dict_next.append(data_dict_next)
                n_agents[idx] = t.n_t
                list_actions.append(t.a_t)
                reward[idx] = t.r_t
                n_agents_next[idx] = t.n_tp1
                done[idx] = t.done
            actions = np.hstack(list_actions)
        else:
            for t in batch:
                data_dict = {'globals': 0, # unused
                             'nodes': t[0],  # np.array of node attributes
                             'edges': t[1],  # np.array of edge attributes
                             # indices of sender nodes for each edge
                             'senders': np.nonzero(t[2])[0],
                             # indices of receiver nodes for each edge
                             'receivers': np.nonzero(t[2])[1]}
                list_data_dict.append(data_dict)
                data_dict_next = {'globals': 0, # unused
                                  'nodes': t[6],  # np.array of node attributes
                                  'edges': t[7],  # np.array of edge attributes
                                  # indices of sender nodes for each edge
                                  'senders': np.nonzero(t[8])[0],
                                  # indices of receiver nodes for each edge
                                  'receivers': np.nonzero(t[8])[1]}
                list_data_dict_next.append(data_dict_next)
            n_agents = np.stack(batch[:,3])
            actions = np.hstack(batch[:,4])
            reward = np.stack(batch[:,5])
            n_agents_next = np.stack(batch[:,9])
            done = np.stack(batch[:,10])
        done_multiplier = -(done - 1)
        
        # [batch_size * (n_agents), dim_action]
        actions_1hot = marl_util.batch_action_int_to_1hot(
            actions, self.dim_action)

        # Convert list of data_dicts into GraphTuple
        graphs_tuple = utils_np.data_dicts_to_graphs_tuple(list_data_dict)
        graphs_tuple_next = utils_np.data_dicts_to_graphs_tuple(list_data_dict_next)

        # if self.noisy_net:
        #     for layer in self.list_nl_target:
        #         layer.reset_noise(tf.float32)

        if self.ddqn:
            # Get argmax_a Q(s_{t+1}, a)
            feed = utils_tf.get_feed_dict(self.graphs_tuple_ph, graphs_tuple_next)
            feed[self.ragged_row_lengths] = n_agents_next
            # [n_1...n_b]
            argmax_actions_next = sess.run(self.argmax_q, feed_dict=feed)
            actions_1hot_next = marl_util.batch_action_int_to_1hot(
                argmax_actions_next, self.dim_action)
            # Evaluate Q_target using argmax actions
            feed = utils_tf.get_feed_dict(self.graphs_tuple_ph, graphs_tuple_next)
            feed.update({self.ragged_row_lengths: n_agents_next,
                         self.actions_1hot: actions_1hot_next})
            q_target = sess.run(self.q_global_target, feed_dict=feed)
        else:
            # Get Q_global target value
            feed = utils_tf.get_feed_dict(self.graphs_tuple_ph, graphs_tuple_next)
            feed[self.ragged_row_lengths] = n_agents_next
            q_target = sess.run(self.q_global_target_max, feed_dict=feed)
        target = (reward + (self.gamma**self.multi_step) * np.squeeze(q_target)
                  * done_multiplier)
        
        feed = utils_tf.get_feed_dict(self.graphs_tuple_ph, graphs_tuple)
        feed.update({self.ragged_row_lengths: n_agents,
                     self.actions_1hot: actions_1hot,
                     self.td_target: target})
        if self.prioritized:
            feed[self.weights] = weights
            _, td_errors = sess.run([self.train_op, self.td_errors],
                                    feed_dict=feed)
        else:
            _ = sess.run(self.train_op, feed_dict=feed)
            td_errors = None

        sess.run(self.list_update_target_ops)

        return td_errors
