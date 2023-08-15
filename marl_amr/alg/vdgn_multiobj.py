"""Value decomposition graph network with Multi-Objective Q-learning (MOQ).

Implementation of single-agent MOQ: https://github.com/RunzheYang/MORL.
"""

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
        self.n_objectives = config.n_objectives
        # Note: when reshaping q output, order must be [dim_action, n_obj]
        self.n_q_i_out = (dim_action * self.n_objectives if
                          self.n_objectives else dim_action)
        # size of set W used in max_{w in W}
        self.n_weights = config.n_weights
        self.beta_init = config.beta_init
        self.beta = config.beta_init
        tau = 1000
        self.beta_expbase = float(np.power(tau*(1.0 - config.beta_init),
                                           1/config.n_episodes))
        self.beta_delta = self.beta_expbase / tau
        super().__init__(config, config_nn, None, dim_action)

    def update_beta(self):
        self.beta += self.beta_delta
        self.beta_delta = ((self.beta - self.beta_init)*self.beta_expbase +
                           self.beta_init - self.beta)

    def create_networks(self):

        # Create a placeholder using a graph as template.
        self.graphs_tuple_ph = utils_tf.placeholders_from_data_dicts(
            [self.sample_data_dict])

        self.ragged_row_lengths = tf.compat.v1.placeholder(
            tf.int32, [None], 'ragged_row_lengths')

        self.actions_1hot = tf.compat.v1.placeholder(tf.float32,
                                           [None, self.dim_action],
                                           'actions_1hot')

        # [batch, n_objectives]
        self.preference = tf.compat.v1.placeholder(
            tf.float32, [None, self.n_objectives], 'preference')

        with tf.compat.v1.variable_scope('q_i_main'):
            # When called by run_actor, shape is
            # [(nodes), n_q_i_out]
            # where n_q_i_out = dim_action * n_objectives
            # When called by train, shape is
            # [batch*(nodes)...batch*(nodes), n_q_i_out]
            # where there are n_weights repeats of batch*(nodes)
            self.q_i, _ = networks.graph_local_q(
                self.graphs_tuple_ph, self.config_nn, self.n_q_i_out)
        with tf.compat.v1.variable_scope('q_i_target'):
            # Q(s_{j+1}, a, w_i) for all j in [batch_size], i in [n_weights]
            # [batch*(nodes)...batch*(nodes), n_q_i_out]
            # where there are n_weights repeats of batch*(nodes)
            self.q_i_target, _ = networks.graph_local_q(
                self.graphs_tuple_ph, self.config_nn, self.n_q_i_out)

        # [batch*(nodes), dim_action]
        self.wq_i = self.multiply_q_i_with_preference(self.q_i)

        # ------------ Get arg_q max_{a,w} w_i Q(s_{j+1}, a, w) ------------- #
        # [batch*(nodes)...batch*(nodes), dim_action, n_weights]
        # where there are n_weights groups of batch*(nodes)
        wq_i_target = self.multiply_q_i_target_preference(self.q_i_target)
        # last dim n_weights is indexed by i, batch*(nodes) is j
        # maximization is done on dim_action and the first n_weights groups
        # [n_weights, batch*(nodes), dim_action, n_weights]
        #    max           j             max         i
        wq_i_target = tf.reshape(
            wq_i_target, [self.n_weights, -1, self.dim_action, self.n_weights])
        # [batch*(nodes), n_weights, dim_action, n_weights]
        #     j              i          max          max
        wq_i_target = tf.transpose(wq_i_target, [1,3,2,0])
        # [batch*(nodes), n_weights, dim_action]
        self.argmax_q_target_w = tf.argmax(wq_i_target, axis=3)
        # [batch*(nodes), n_weights, dim_action]
        wq_i_target_max_w = tf.reduce_max(wq_i_target, axis=3)
        # [batch*(nodes), n_weights]
        self.argmax_q_target_w_a = tf.argmax(wq_i_target_max_w, axis=2)

        # [n_weights, batch*(nodes), dim_action, n_obj]
        #     max   ,      j       ,   max     , ignore
        q_i_target = tf.reshape(self.q_i_target,
                                [self.n_weights, -1, self.dim_action, self.n_objectives])
        # [batch*(nodes), dim_action, n_weights, n_obj]
        #      j        ,     max   ,    max   , ignore
        q_i_target = tf.transpose(q_i_target, [1,2,0,3])
        # Duplicate to get the i dimension
        # [n_weights groups of batch*(nodes), dim_action, n_weights, n_obj]
        q_i_target = tf.tile(q_i_target, [self.n_weights, 1,1,1])
        # [n_weights, batch*(nodes), dim_action, n_weights, n_obj]
        q_i_target = tf.reshape(q_i_target, [self.n_weights, -1, self.dim_action,
                                             self.n_weights, self.n_objectives])
        # [batch*(nodes), n_weights, dim_action, n_weights, n_obj]
        #          j    ,    i     ,    max    ,    max   , leave
        q_i_target = tf.transpose(q_i_target, [1,0,2,3,4])
        # [batch*(nodes), n_weights, dim_action, n_obj]
        #          j    ,    i     ,    max    , leave
        q_i_target = tf.gather(q_i_target,
                               tf.expand_dims(self.argmax_q_target_w, axis=3),
                               axis=3, batch_dims=-1)
        # [batch*(nodes), n_weights, n_obj]
        q_i_target = tf.gather(q_i_target,
                               tf.expand_dims(self.argmax_q_target_w_a, axis=2),
                               axis=2, batch_dims=-1)
        # [batch, (nodes), n_weights, n_obj]
        q_i_target_ragged = tf.RaggedTensor.from_row_lengths(
            q_i_target, row_lengths=self.ragged_row_lengths)
        # [batch, n_weights, n_obj]
        self.q_global_target = tf.reduce_sum(q_i_target_ragged, axis=1)
        # -------------------------------------------------------------------- #

        # Used in run_actor()
        # [batch*(nodes)]
        self.argmax_q = tf.argmax(self.wq_i, axis=1)

        # -------------------- Use by train step only -------------------- #
        # [n_weights groups of batch*(nodes), dim_action, n_obj]
        q_i_rs = tf.reshape(self.q_i, [-1, self.dim_action, self.n_objectives])
        # [n_weights groups of b*(n), n_obj, |A|]
        q_i_rs = tf.transpose(q_i_rs, [0,2,1])
        # [n_weights groups of b*(n), |A|]
        actions_1hot_tile = tf.tile(self.actions_1hot, [self.n_weights, 1])
        # [n_weights groups of b*(n), 1, |A|]
        actions_1hot_expand = tf.expand_dims(actions_1hot_tile, 1)
        # tf.multiply broadcasts along axis 1
        # [n_weights groups of batch*(nodes), n_obj]
        q_i_selected = tf.reduce_sum(tf.multiply(
            q_i_rs, actions_1hot_expand), axis=2)
        # [n_weights, b*(n), n_obj]
        q_i_selected = tf.reshape(q_i_selected,
                                  [self.n_weights, -1, self.n_objectives])
        # [b*(n), n_weights, n_obj]
        q_i_selected = tf.transpose(q_i_selected, [1,0,2])
        # [batch, (nodes), n_weights, n_obj]
        q_i_ragged = tf.RaggedTensor.from_row_lengths(
            q_i_selected, row_lengths=self.ragged_row_lengths)
        # Without this, an error appears in create_train_op()
        q_i_ragged = q_i_ragged.to_tensor(default_value = 0)
        # For each graph, sum selected q-value over all nodes
        # [batch, n_weights, n_obj]
        self.q_global = tf.reduce_sum(q_i_ragged, axis=1)

    def multiply_q_i_with_preference(self, q_i):
        """Computes w^T Q^i for each node, for each group of nodes in batch.

        Note that each group is a state, which may be drawn from the replay buffer
        and hence each group has its own preference.
        self.preference: [batch, n_objectives]
        self.ragged_row_lengths: [batch], number of nodes in each state

        Args:
            q_i: [batch*(nodes), dim_action*n_objectives]

        Returns:
            [batch*(nodes), dim_action]
        """
        # [batch*(nodes), dim_action, n_objectives]
        q_i = tf.reshape(q_i, [-1, self.dim_action, self.n_objectives])
        # Repeat each preference[i] by ragged_row_lengths[i]
        # [batch*(nodes), n_objectives]
        preference_repeat = utils_tf.repeat(
            self.preference, self.ragged_row_lengths, axis=0)
        # [batch*(nodes), n_objectives, 1]
        preference_repeat = tf.expand_dims(preference_repeat, axis=2)
        # [batch*(nodes), dim_action, 1]
        w_q_i = tf.matmul(q_i, preference_repeat)
        # [batch*(nodes), dim_action]
        w_q_i = tf.reshape(w_q_i, [-1, self.dim_action])

        return w_q_i

    def multiply_q_i_target_preference(self, q_i_target):
        """Computes w^T Q^i for each node, for each group of nodes in batch.

        Note that each group is a state, which may be drawn from the replay buffer
        and hence each group has its own preference.
        self.preference: [n_weights, n_objectives]
        self.ragged_row_lengths: [batch], number of nodes in each state

        Args:
            q_i_target: [batch*(nodes)...batch*(nodes), dim_action*n_objectives]
                where there are n_weights groups of batch*(nodes)

        Returns:
            [batch*(nodes), dim_action, n_weights]
        """
        #          to be maxed
        # [n_weights groups of batch*(nodes), dim_action, n_obj]
        q_i_target = tf.reshape(q_i_target, [-1, self.dim_action, self.n_objectives])

        # Get preference into shape [q_i_target.shape[0], n_objectives, n_weights]
        total_batch_size = tf.reduce_sum(self.ragged_row_lengths) # batch*(nodes)
        num_tiles = total_batch_size * self.n_weights
        # [n_obj, n_weights]
        preference_t = tf.transpose(self.preference, [1,0])
        # [num_tiles repeats of n_obj, n_weights]
        preference_tiled = tf.tile(preference_t, [num_tiles, 1])
        # [n_weights groups of batch*(nodes), n_obj, n_weights]
        preference_rs = tf.reshape(preference_tiled,
                                   [num_tiles, self.n_objectives, self.n_weights])
        # [n_weights groups of batch*(nodes), dim_action, n_weights]
        w_q_i_target = tf.matmul(q_i_target, preference_rs)

        return w_q_i_target

    def run_actor(self, graph, epsilon, sess, preference):
        """Gets actions for all agents as a batch.
        
        Args:
            graph: tuple of (nodes, edges, adj_matrix)
            epsilon: float, epsilon-greedy exploration
            sess: TF session
            preference: 1D np.array

        Returns: np.array of discrete actions
        """
        n_agents = len(graph[0])
        # Concatenate preference with base obs of all nodes
        nodes = np.array([np.concatenate((obs, preference))
                          for obs in graph[0]])
        data_dict = {'globals': 0, # unused
                     'nodes': nodes.astype(np.float32),  # np.array of node attributes
                     'edges': graph[1].astype(np.float32),  # np.array of edge attributes
                     # indices of sender nodes for each edge
                     'senders': np.nonzero(graph[2])[0].astype(np.int32),
                     # indices of receiver nodes for each edge
                     'receivers': np.nonzero(graph[2])[1].astype(np.int32)}
        graph_tuple = utils_np.data_dicts_to_graphs_tuple([data_dict])
        feed_dict = utils_tf.get_feed_dict(
            self.graphs_tuple_ph, utils_np.data_dicts_to_graphs_tuple([data_dict]))
        feed_dict[self.preference] = np.array([preference])
        feed_dict[self.ragged_row_lengths] = np.array([n_agents])

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

    def create_train_op(self):
        """Defines loss and creates train op."""
        # y_{ij} = r_j + \gamma arg_Q max_{a,w} w_i^T Q(s_{j+1},a,w)
        # where j = 1...batch, i = 1...n_weights
        # so each group consists of w_i i=1...n_weights
        # [batch groups of n_weights, n_obj]
        self.td_target = tf.compat.v1.placeholder(
            tf.float32, [None, self.n_objectives], 'td_target')
        
        # self.preference: [n_weights, n_obj]
        # [batch groups of n_weights, n_obj]
        self.preference_repeat = tf.tile(self.preference, [self.batch_size, 1])

        # w^T y
        # [batch groups of n_weights]
        self.scalar_td_target = tf.reduce_sum(
            tf.multiply(self.td_target, self.preference_repeat), axis=1)
        
        # [batch groups of n_weights, n_obj]
        q_global = tf.reshape(self.q_global, [-1, self.n_objectives])

        # w^T Q
        # [batch groups of n_weights]
        self.scalarized_q = tf.reduce_sum(
            tf.multiply(q_global, self.preference_repeat), axis=1)

        # L^b(theta) = E_{s,a,w}[|w^T(y - Q)|]
        self.loss_auxiliary = tf.reduce_mean(tf.math.abs(
            self.scalar_td_target - self.scalarized_q))

        # L^a(theta) = E_{s,a,w}[||y - Q||^2_2]
        self.loss_main = tf.reduce_mean(tf.reduce_sum(
            tf.square(self.td_target - q_global), axis=1))

        self.loss = (1-self.beta)*self.loss_main + self.beta*self.loss_auxiliary

        self.opt = tf.compat.v1.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.loss)

    def train(self, batch, sess):
        """One training step.

        Args: 
            batch: np.array of transitions, each transition is an np.array of
                   (0. node array, 1. edge array, 2. adj_matrix, 
                    3. n_agents, 4. actions, 5. reward, 
                    6. next node array, 7. next edge array, 8. next adj_matrix,
                    9. n_agents next, 10. done)
            sess: TF session
        """
        # sample n_weights preferences
        # [n_weights, n_obj]
        preferences = np.random.randn(self.n_weights, self.n_objectives)
        preferences = (np.abs(preferences) /
                       np.linalg.norm(preferences, 1, axis=1, keepdims=True))

        # create all pairs of state j+1 with preference w_i
        # ordering is [b_1...b_batch, ... ,b_1...b_batch]
        # where there are n_weights groups of batch size
        list_data_dict = []
        list_data_dict_next = []
        for i in range(self.n_weights):
            for t in batch:
                nodes = np.array([np.concatenate((obs, preferences[i]))
                                       for obs in t[0]])
                nodes_next = np.array([np.concatenate((obs, preferences[i]))
                                       for obs in t[6]])
                data_dict = {'globals': 0,
                             'nodes': nodes,
                             'edges': t[1],
                             # indices of sender nodes for each edge
                             'senders': np.nonzero(t[2])[0],
                             # indices of receiver nodes for each edge
                             'receivers': np.nonzero(t[2])[1]}
                list_data_dict.append(data_dict)
                data_dict_next = {'globals': 0,
                                  'nodes': nodes_next,
                                  'edges': t[7],
                                  'senders': np.nonzero(t[8])[0],
                                  'receivers': np.nonzero(t[8])[1]}
                list_data_dict_next.append(data_dict_next)

        # Convert list of data_dicts into GraphTuple
        graphs_tuple = utils_np.data_dicts_to_graphs_tuple(list_data_dict)
        graphs_tuple_next = utils_np.data_dicts_to_graphs_tuple(list_data_dict_next)

        n_agents = np.stack(batch[:,3]) # [batch]
        actions = np.hstack(batch[:,4]) # [batch]
        reward = np.stack(batch[:,5]) # [batch, n_obj]
        n_agents_next = np.stack(batch[:,9]) # [batch]
        done = np.stack(batch[:,10]) # [batch]

        # [batch_size * (n_agents), dim_action]
        actions_1hot = marl_util.batch_action_int_to_1hot(
            actions, self.dim_action)

        # ----- compute r_j + gamma arg_q max_{a,w} w_i Q(s_{j+1}, a, w) ----- #
        # Using target network
        # get arg_Q max_{a,w} w_i Q(s_{j+1}, a, w)
        # for each j = 1...batch, i=1...n_weights
        feed = utils_tf.get_feed_dict(self.graphs_tuple_ph, graphs_tuple_next)
        feed.update({self.ragged_row_lengths: n_agents_next,
                     self.preference: preferences})
        # [batch, n_weights, n_obj]
        q_global_target = sess.run(self.q_global_target, feed_dict=feed)
        # [batch groups of n_weights, n_obj]
        # q_global_target = np.reshape(q_global_target, [-1, self.n_objectives])

        done_multiplier = -(done - 1)
        # [batch groups of self.n_weights*self.n_objectives]
        done_repeat = np.repeat(done_multiplier, self.n_weights*self.n_objectives)
        # All values at done_multiplier[i, :, :] are the same
        # [batch, n_weights, n_obj]
        done_multiplier = np.reshape(done_repeat,
            [self.batch_size, self.n_weights, self.n_objectives])

        # [batch groups of n_weights, n_obj]
        reward_repeat = np.repeat(reward, self.n_weights, axis=0)
        reward = np.reshape(reward_repeat,
                            [self.batch_size, self.n_weights, self.n_objectives])

        # y_{ij} = r_j + \gamma arg_Q max_{a,w} w_i^T Q(s_{j+1},a,w)
        # where j = 1...batch, i = 1...n_weights
        # [batch, n_weights, n_obj]
        target = reward + self.gamma * np.squeeze(q_global_target) * done_multiplier
        # [batch groups of n_weights, n_obj]
        target = np.reshape(target, [-1, self.n_objectives])
        # ---------------------------------------------------------------- #
        
        feed = utils_tf.get_feed_dict(self.graphs_tuple_ph, graphs_tuple)
        feed.update({self.actions_1hot: actions_1hot,
                     self.ragged_row_lengths: n_agents,
                     self.preference: preferences,
                     self.td_target: target})
        _ = sess.run(self.train_op, feed_dict=feed)

        sess.run(self.list_update_target_ops)
