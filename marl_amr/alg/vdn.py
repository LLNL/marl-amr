"""Value decomposition network (Sunehag et al. 2017)."""

import numpy as np
import tensorflow as tf

from marl_amr.alg import networks

from marl_amr.alg.utils import marl_util


class Alg(object):

    def __init__(self, config, config_nn, dim_obs, dim_action):
        """
        Args:
            config: ConfigDict object with algorithm hyperparameters
            config_nn: ConfigDict object for neural network
            dim_obs: dimension of observation, either int or list
            dim_action: integer dimension of discrete action space
        """
        self.config = config
        self.config_nn = config_nn
        self.batch_size = config.batch_size
        self.explore_type = config.explore_type
        self.gamma = config.gamma
        self.lr = config.lr
        self.name = config.name
        self.tau = config.tau

        self.dim_obs = dim_obs
        self.dim_action = dim_action

        self.ddqn = config.ddqn
        self.dueling = config.dueling
        self.multi_step = config.multi_step if config.multi_step else 1
        self.noisy_net = config.noisy_net
        self.prioritized = config.prioritized_replay

        self.create_networks()
        self.create_target_ops()
        self.create_weight_setter_ops()
        self.create_train_op()

    def create_networks(self):

        if isinstance(self.dim_obs, int):
            self.obs = tf.compat.v1.placeholder(
                tf.float32, [None, self.dim_obs], 'obs')
            model = networks.mlp
        elif isinstance(self.dim_obs, list):
            self.obs = tf.placeholder(tf.float32, [None]+self.dim_obs, 'obs')
            model = networks.conv_mlp

        with tf.compat.v1.variable_scope('q_i_main'):
            # Due to parameter-sharing, output is
            # [batch*n_agents, dim_action]
            self.q_i = model(self.obs, n_outputs=self.dim_action,
                             config=self.config_nn)
        with tf.compat.v1.variable_scope('q_i_target'):
            self.q_i_target = model(
                self.obs, n_outputs=self.dim_action, config=self.config_nn)

        self.argmax_q = tf.argmax(self.q_i, axis=1)

        self.actions_1hot = tf.compat.v1.placeholder(tf.float32,
                                           [None, self.dim_action],
                                           'actions_1hot')
        # [batch*n_agents]
        self.q_i_selected = tf.reduce_sum(tf.multiply(
            self.q_i, self.actions_1hot), axis=1)
        # Sum over agents, for each batch entry
        self.ragged_row_lengths = tf.compat.v1.placeholder(
            tf.int32, [None], 'ragged_row_lengths')
        q_i_ragged = tf.RaggedTensor.from_row_lengths(
            self.q_i_selected, row_lengths=self.ragged_row_lengths)
        self.q_global = tf.reduce_sum(q_i_ragged, axis=1)

        self.q_i_target_max = tf.math.reduce_max(
            self.q_i_target, axis=1)
        q_i_target_ragged = tf.RaggedTensor.from_row_lengths(
            self.q_i_target_max, row_lengths=self.ragged_row_lengths)
        self.q_global_target = tf.reduce_sum(q_i_target_ragged, axis=1)

    def create_target_ops(self):

        self.list_initialize_target_ops = []
        self.list_update_target_ops = []

        # Updates to target q_i
        self.q_i_var = tf.compat.v1.trainable_variables('q_i_main')
        self.q_i_target_var = tf.compat.v1.trainable_variables('q_i_target')
        for idx, var in enumerate(self.q_i_target_var):
            self.list_initialize_target_ops.append(
                var.assign(self.q_i_var[idx]))
            self.list_update_target_ops.append(
                var.assign(self.tau*self.q_i_var[idx] + (1-self.tau)*var))

    def create_weight_setter_ops(self):
        """Creates placeholders and ops for setting weights externally."""
        self.list_q_i_ph = []
        self.list_set_q_ops = []
        for var in self.q_i_var:
            ph = tf.placeholder(var.dtype)
            self.list_q_i_ph.append(ph)
            self.list_set_q_ops.append(var.assign(ph))

        self.list_q_i_target_ph = []
        self.list_set_q_target_ops = []
        for var in self.q_i_target_var:
            ph = tf.placeholder(var.dtype)
            self.list_q_i_target_ph.append(ph)
            self.list_set_q_target_ops.append(var.assign(ph))

    def run_actor(self, obs, epsilon, sess):
        """Gets actions for all agents as a batch.
        
        Args:
            list_obs: list of observations
            epsilon: float, epsilon-greedy exploration
            sess: TF session

        Returns: np.array of discrete actions
        """
        n_agents = len(obs)
        feed = {self.obs: np.array(obs)}
        if self.explore_type == 'global':
            if np.random.rand(1) < epsilon:
                return np.random.randint(0, self.dim_action, n_agents)
            else:
                return sess.run(self.argmax_q, feed_dict=feed)
        elif self.explore_type == 'independent':
            actions_argmax = sess.run(self.argmax_q, feed_dict=feed)
            actions = np.zeros(n_agents, dtype=int)
            for idx in range(n_agents):
                if np.random.rand(1) < epsilon:
                    actions[idx] = np.random.randint(0, self.dim_action)
                else:
                    actions[idx] = actions_argmax[idx]

        return actions

    def create_train_op(self):
        # TD target is computed in train() using mixer_target
        self.td_target = tf.compat.v1.placeholder(
            tf.float32, [None], 'td_target')
        self.td_errors = self.td_target - tf.squeeze(self.q_global)
        losses = tf.square(self.td_errors)
        if self.prioritized:
            self.weights = tf.compat.v1.placeholder(
                tf.float32, [None], 'weights')
            self.loss = tf.reduce_mean(tf.multiply(losses, self.weights))
        else:
            self.loss = tf.reduce_mean(losses)

        self.opt = tf.compat.v1.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.loss)

    def train(self, batch, sess):
        """One training step.

        Args: 
            batch: np.array of transitions, each transition is an np.array of
                   (list of obs, n_agents, actions, reward, list of next obs,
                   n_agents next, done)
            sess: TF session
        """
        (obs, n_agents, actions, reward, obs_next,
         n_agents_next, done) = marl_util.unpack_batch_local(batch)

        # [batch_size * (n_agents), dim_action]
        actions_1hot = marl_util.batch_action_int_to_1hot(
            actions, self.dim_action)

        # Get Q_global target value
        feed = {self.obs: obs_next,
                self.ragged_row_lengths: n_agents_next}
        q_global_target = sess.run(self.q_global_target, feed_dict=feed)

        done_multiplier = -(done - 1)
        target = reward + self.gamma * np.squeeze(q_global_target) * done_multiplier

        feed = {self.obs: obs,
                self.ragged_row_lengths: n_agents,
                self.actions_1hot: actions_1hot,
                self.td_target: target}
        _ = sess.run(self.train_op, feed_dict=feed)

        sess.run(self.list_update_target_ops)
