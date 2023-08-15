"""Extension of graph_nets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_tf

import numpy as np
import sonnet as snt
import tensorflow as tf


NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE


class CustomEdgeBlock(blocks.EdgeBlock):
    """Edge block.

    Modification of graph_nets.blocks.EdgeBlock.
    """
    def __init__(self,
                 edge_model_fn,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True,
                 name="edge_block"):
        """See graph_nets.blocks.EdgeBlock."""
        super(CustomEdgeBlock, self).__init__(
            edge_model_fn,
            use_edges,
            use_receiver_nodes,
            use_sender_nodes,
            use_globals,
            name
        )

    def _build(self, graph):
        """Connects the edge block.

        Modification from original: instead of concatenating global attributes
        with the rest into collected_nodes, use it as a separate input
        to self._node_model.

        Args:
            graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
                features (if `use_edges` is `True`), individual nodes features (if
                `use_receiver_nodes` or `use_sender_nodes` is `True`) and per graph
                globals (if `use_globals` is `True`) should be concatenable on the last
                axis.

        Returns:
            An output `graphs.GraphsTuple` with updated edges.

        Raises:
            ValueError: If `graph` does not have non-`None` receivers and senders, or
                if `graph` has `None` fields incompatible with the selected `use_edges`,
                `use_receiver_nodes`, `use_sender_nodes`, or `use_globals` options.
        """
        blocks._validate_graph(
            graph, (SENDERS, RECEIVERS, N_EDGE), " when using an EdgeBlock")
    
        edges_to_collect = []
    
        if self._use_edges:
            blocks._validate_graph(graph, (EDGES,), "when use_edges == True")
            edges_to_collect.append(graph.edges)
    
        if self._use_receiver_nodes:
            edges_to_collect.append(
                blocks.broadcast_receiver_nodes_to_edges(graph))
    
        if self._use_sender_nodes:
            edges_to_collect.append(
                blocks.broadcast_sender_nodes_to_edges(graph))
    
        if self._use_globals:
            edges_to_collect.append(broadcast_globals_to_edges(graph))
  
        collected_edges = tf.concat(edges_to_collect, axis=-1)
        # graph.n_edge is a sequence of number of edges in each graph
        updated_edges = self._edge_model(collected_edges, graph.n_edge)
  
        return graph.replace(edges=updated_edges)


class CustomNodeBlock(blocks.NodeBlock):
    """Node block.

    Modification of graph_nets.blocks.NodeBlock.
    """
    def __init__(self,
                 node_model_fn,
                 use_received_edges=True,
                 use_sent_edges=False,
                 use_nodes=True,
                 use_globals=True,
                 received_edges_reducer=tf.unsorted_segment_sum,
                 sent_edges_reducer=tf.unsorted_segment_sum,
                 name="node_block"):
        """See graph_nets.blocks.NodeBlock."""
        super(CustomNodeBlock, self).__init__(
            node_model_fn,
            use_received_edges,
            use_sent_edges,
            use_nodes,
            use_globals,
            received_edges_reducer,
            sent_edges_reducer,
            name)

    def _build(self, graph):
        """Connects the node block.

        Modification from original: instead of concatenating global attributes
        with the rest into collected_nodes, use it as a separate input
        to self._node_model.

        Args:
          graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
            features (if `use_received_edges` or `use_sent_edges` is `True`),
            individual nodes features (if `use_nodes` is True) and per graph globals
            (if `use_globals` is `True`) should be concatenable on the last axis.
          node_model_kwargs: Optional keyword arguments to pass to the `node_model`.

        Returns:
          An output `graphs.GraphsTuple` with updated nodes.
        """
        nodes_to_collect = []

        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))

        if self._use_nodes:
            blocks._validate_graph(graph, (NODES,), "when use_nodes == True")
            nodes_to_collect.append(graph.nodes)

        if self._use_globals:
            nodes_to_collect.append(broadcast_globals_to_nodes(graph))

        collected_nodes = tf.concat(nodes_to_collect, axis=-1)
        # graph.n_node is a sequence of number of nodes in each graph.
        updated_nodes = self._node_model(collected_nodes, graph.n_node)

        return graph.replace(nodes=updated_nodes)


class CustomInteractionNetwork(snt.AbstractModule):
    """Implementation of an Interaction Network.

    Modification of graph_nets.modules.InteractionNetwork.
    Uses CustomNodeBlock instead of NodeBlock.
    Uses CustomEdgeBlock instead of EdgeBlock.
    """

    def __init__(self,
                 edge_model_fn,
                 node_model_fn,
                 reducer=tf.unsorted_segment_sum,
                 name="interaction_network"):
        """Initializes the InteractionNetwork module.

        Args:
            edge_model_fn: A callable that will be passed to `EdgeBlock` to perform
                per-edge computations. The callable must return a Sonnet module (or
                equivalent; see `blocks.EdgeBlock` for details), and the shape of the
                output of this module must match the one of the input nodes, but for the
                first and last axis.
            node_model_fn: A callable that will be passed to `NodeBlock` to perform
                per-node computations. The callable must return a Sonnet module (or
                equivalent; see `blocks.NodeBlock` for details).
            reducer: Reducer to be used by NodeBlock to aggregate edges. Defaults to
                tf.math.unsorted_segment_sum.
            name: The module name.
        """
        super(CustomInteractionNetwork, self).__init__(name=name)

        with self._enter_variable_scope():
            self._edge_block = CustomEdgeBlock(
                edge_model_fn=edge_model_fn, use_globals=False)
            self._node_block = CustomNodeBlock(
                node_model_fn=node_model_fn,
                use_sent_edges=False,
                use_globals=False,
                received_edges_reducer=reducer)

    def _build(self, graph):
        """Connects the InterationNetwork.
        Args:
            graph: A `graphs.GraphsTuple` containing `Tensor`s. `graph.globals` can be
                `None`. The features of each node and edge of `graph` must be
                concatenable on the last axis (i.e., the shapes of `graph.nodes` and
                `graph.edges` must match but for their first and last axis).

        Returns:
            An output `graphs.GraphsTuple` with updated edges and nodes.

        Raises:
            ValueError: If any of `graph.nodes`, `graph.edges`, `graph.receivers` or
                `graph.senders` is `None`.
        """
        return self._node_block(self._edge_block(graph))


def CustomLinear(inputs, repeats, w, b):
    """Linear function using external w and b.

    Purpose of this module is to store weights and bias, which are
    trainable variables created by a hypernetwork, for use by graphnets.

    Hypernetwork generates a unique w[i] and b[i] for each graph i.
    However, this module will be applied to the nodes or edges of all graphs.
    Hence, duplicate w[i] and b[i] by the number of nodes or edges in graph i,
    and use the batch-matrix-multiply ability of tf.matmul to pair the correct
    weight and bias to each node.

    batch size [batch] = b
    input's first dimension consists of b groups of n_i, where
    n_i is the number of nodes or edges in graph i.

    Args:
        inputs: tf.Tensor of shape [n_1...n_b, input_dim]
            e.g.: If called by mixer layer 1 edge block,
            input_dim is 2 + dim_edge since initial nodes hold scalar q^i.
        repeats: sequence of length `batch`, repeats[i] is the number of
            nodes in batch i (i.e., graph i)
        w: tf.Tensor of shape [batch, input dim, output dim]
        b: tf.Tensor of shape [batch, output dim]
    """
    # [n_1...n_b, 1, input dim]
    inputs = tf.expand_dims(inputs, axis=1)

    # [n_1...n_b, input dim, output dim]
    w_repeat = utils_tf.repeat(w, repeats, axis=0)
    # [n_1...n_b, output_dim]
    b_repeat = utils_tf.repeat(b, repeats, axis=0)

    # [n_1...n_b, 1, output_dim]
    outputs = tf.matmul(inputs, w_repeat)
    # If axis is not specified, then squeeze produces shape <unknown>
    # [n_1...n_b, output_dim]
    squeezed = tf.squeeze(outputs, axis=1)
    # [n_1...n_b, output_dim]
    outputs = tf.add(squeezed, b_repeat)

    return outputs


def NoisyLinear(inputs, n_out, std_init=0.5):
    """Factorized NoisyNet (Fortunato et al. 2017).

    y = (w_mu + w_sigma * w_noise)x + b_mu + b_sigma * b_noise

    Args:
    inputs: tf.Tensor with shape [batch, n_in]
    n_out: int
    std_init: float

    Returns:
    tf.Tensor
    """
    n_in = inputs.get_shape().as_list()[-1]
    mu_range = 1/np.sqrt(n_in)
    w_mu_init = tf.initializers.random_uniform(
        minval=-mu_range, maxval=mu_range)
    w_sigma_init = tf.initializers.constant(std_init/np.sqrt(n_in))
    b_mu_init = tf.initializers.random_uniform(
        minval=-mu_range, maxval=mu_range)
    b_sigma_init = tf.initializers.constant(std_init/np.sqrt(n_in))

    w_shape = [n_in, n_out]
    w_mu = tf.compat.v1.get_variable('w_mu', w_shape, initializer=w_mu_init)
    w_sigma = tf.compat.v1.get_variable('w_sigma', w_shape,
                                        initializer=w_sigma_init)

    b_shape = [1, n_out]
    b_mu = tf.compat.v1.get_variable('b_mu', b_shape, initializer=b_mu_init)
    b_sigma = tf.compat.v1.get_variable('b_sigma', b_shape,
                                        initializer=b_sigma_init)

    epsilon_in = tf.random.normal(shape=n_in)
    epsilon_out = tf.random.norm(shape=n_out)
    def scale_noise(epsilon):
        """f(x) = sgn(x)sqrt(|x|)"""
        return tf.multiply(tf.math.sign(epsilon),
                           tf.math.sqrt(tf.math.abs(epsilon)))
    f_epsilon_in = scale_noise(epsilon_in)
    f_epsilon_out = scale_noise(epsilon_out)
    w_epsilon = tf.tensordot(f_epsilon_in, f_epsilon_out, axes=0)
    b_epsilon = epsilon_out

    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    b = b_mu + tf.mulitply(b_sigma, b_epsilon)

    return tf.add(tf.matmul(inputs, w), b)


class NoisyLinear(snt.Linear):
    """Factorized NoisyNet (Fortunato et al. 2017)."""

    def __init__(self, output_size, mode='train', name="noisy_linear",
                 std_init=0.5):
        super(NoisyLinear, self).__init__(output_size, name=name)
        self.mode = mode
        self.std_init = std_init

    def reset_parameters(self, n_in, n_out, dtype):

        mu_range = 1/np.sqrt(self._input_size)
        self._initializers['w_mu_init'] = tf.initializers.random_uniform(
            minval=-mu_range, maxval=mu_range, dtype=dtype)
        self._initializers['w_sigma_init'] = tf.initializers.constant(
            self.std_init/np.sqrt(self._input_size), dtype=dtype)
        self._initializers['b_mu_init'] = tf.initializers.random_uniform(
            minval=-mu_range, maxval=mu_range, dtype=dtype)
        self._initializers['b_sigma_init'] = tf.initializers.constant(
            self.std_init/np.sqrt(self._input_size), dtype=dtype)

        w_shape = [self._input_size, n_out]
        self.w_mu = tf.compat.v1.get_variable(
            'w_mu', w_shape, dtype, self._initializers['w_mu_init'])
        self.w_sigma = tf.compat.v1.get_variable(
            'w_sigma', w_shape, dtype, self._initializers['w_sigma_init'])

        b_shape = [1, n_out]
        self.b_mu = tf.compat.v1.get_variable(
            'b_mu', b_shape, dtype, self._initializers['b_mu_init'])
        self.b_sigma = tf.compat.v1.get_variable(
            'b_sigma', b_shape, dtype, self._initializers['b_sigma_init'])

    def scale_noise(self, epsilon):
        """f(x) = sgn(x)sqrt(|x|)"""
        return tf.multiply(tf.math.sign(epsilon),
                           tf.math.sqrt(tf.math.abs(epsilon)))

    def reset_noise(self, dtype):
        epsilon_in = tf.random.normal(shape=[self._input_size], dtype=dtype)
        epsilon_out = tf.random.normal(shape=[self._output_size], dtype=dtype)
        f_epsilon_in = self.scale_noise(epsilon_in)
        f_epsilon_out = self.scale_noise(epsilon_out)
        self.w_epsilon = tf.tensordot(f_epsilon_in, f_epsilon_out, axes=0)
        self.b_epsilon = epsilon_out

    def _build(self, inputs):
        """Connects the Linear module into the graph, with input Tensor `inputs`.

        If this is not the first time the module has been connected to the graph,
        the Tensor provided here must have the same final dimension, in order for
        the existing variables to be the correct size for the multiplication. The
        batch size may differ for each connection.

        Args:
            inputs: A 2D Tensor of size [batch_size, input_size].

        Returns:
            A 2D Tensor of size [batch_size, output_size].

        Raises:
            base.IncompatibleShapeError: If the input is not a 2-D `Tensor` with
                the size of the second dimension specified.
            base.IncompatibleShapeError: If reconnecting an already connected module
                into the graph, and the shape of the input is not compatible with
                previous inputs.
        """
        input_shape = tuple(inputs.get_shape().as_list())

        if len(input_shape) != 2:
            raise base.IncompatibleShapeError(
                "{}: rank of shape must be 2 not: {}".format(
                    self.scope_name, len(input_shape)))

        if input_shape[1] is None:
            raise base.IncompatibleShapeError(
                "{}: Input size must be specified at module build time".format(
                    self.scope_name))

        if self._input_shape is not None and input_shape[1] != self._input_shape[1]:
            raise base.IncompatibleShapeError(
                "{}: Input shape must be [batch_size, {}] not: [batch_size, {}]"
                .format(self.scope_name, self._input_shape[1], input_shape[1]))

        self._input_shape = input_shape
        dtype = inputs.dtype

        self._input_size = input_shape[-1]
        self.reset_parameters(self._input_size, self._output_size, dtype)
        self.reset_noise(dtype)

        if self.mode == 'train':
            self._w = self.w_mu + tf.multiply(self.w_sigma, self.w_epsilon)
            self._b = self.b_mu + tf.multiply(self.b_sigma, self.b_epsilon)
        else:
            self._w = self.w_mu
            self._b = self.b_mu

        outputs = tf.matmul(inputs, self._w) + self._b

        return outputs


class CustomGraphAttentionNetwork(snt.AbstractModule):
    """Implementation of a Graph Network with multihead self-attention.

    Edge update: 
    e^{k'} = (\alpha^{e}(v^{r_k}, v_{s_l}), \beta^e(v^{s_k}, e^k))
           = (a^k', b^k')
    where
    \alpha^{e}(v^{r_k}, v_{s_l}) = exp(NN_query(v^{r_k})^T NN_key(v^{s_k}))
    and
    \beta^e(v^{s_k}, e^k)) = NN_value(v^{s_k}, e^k)

    Aggregation of edges via softmax attention:
    \bar{e^i'} = \rho^{e\rightarrow v}(E^i')
               = \frac{1}{ \sum_{k: r_k=i} a^k'} \sum_{k: r_k=i} a^k' b^k'

    Output
    v^i' = NN_out(concat(\bar{e^i'_1},\dotsc,\bar{e^i'_h}))
    where h = n_heads
    """
    def __init__(self,
                 node_model_fn,
                 output_model_fn,
                 use_edges=False,
                 d_model=512,
                 n_heads=8,
                 name="custom_graph_attention_network"):
        """Initializes the InteractionNetwork module.

        Args:
            node_model_fn: A callable that will be passed to `NodeBlock` to perform
                per-node computations. The callable must return a Sonnet module (or
                equivalent; see `blocks.NodeBlock` for details).
            d_model: (int) Q,K,V have shape [batch, d_model]
            n_heads: (int) number of heads in multi-head attention
            name: The module name.
        """
        super(CustomGraphAttentionNetwork, self).__init__(name=name)

        self.d_model = d_model
        self.n_heads = n_heads
        self.use_edges = use_edges
        assert self.d_model % self.n_heads == 0
        self.d_k = self.d_v = int(self.d_model / self.n_heads)
        if use_edges:
            self._attention = SelfAttention() # use definition in this file
        else:
            self._attention = modules.SelfAttention()

        with self._enter_variable_scope():
            self._value_block = blocks.NodeBlock(
                node_model_fn=node_model_fn,
                use_received_edges=False,
                # previously True, which means all outgoing edges of node are
                # aggregated to affect the node representation
                use_sent_edges=(False if use_edges else True),
                use_nodes=True,
                use_globals=False)
            self._query_block = blocks.NodeBlock(
                node_model_fn=node_model_fn,
                use_received_edges=False,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False)
            self._key_block = blocks.NodeBlock(
                node_model_fn=node_model_fn,
                use_received_edges=False,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False)
            self._output_block = blocks.NodeBlock(output_model_fn, False, False,
                                                  True, False)
            if use_edges:
                self._edge_block = blocks.EdgeBlock(
                    edge_model_fn=node_model_fn, # intentional, can use the same
                    use_receiver_nodes=False,
                    use_sender_nodes=False,
                    use_globals=False)

    def _build(self, graph):
        """Connects the InterationNetwork.
        Args:
            graph: A `graphs.GraphsTuple` containing `Tensor`s. `graph.globals` can be
                `None`. The features of each node and edge of `graph` must be
                concatenable on the last axis (i.e., the shapes of `graph.nodes` and
                `graph.edges` must match but for their first and last axis).

        Returns:
            An output `graphs.GraphsTuple` with updated edges and nodes.

        Raises:
            ValueError: If any of `graph.nodes`, `graph.edges`, `graph.receivers` or
                `graph.senders` is `None`.
        """
        values = tf.reshape(self._value_block(graph).nodes,
                            [-1, self.n_heads, self.d_v])
        keys = tf.reshape(self._key_block(graph).nodes,
                            [-1, self.n_heads, self.d_k])
        queries = tf.reshape(self._query_block(graph).nodes,
                            [-1, self.n_heads, self.d_k])

        if self.use_edges:
            edges = tf.reshape(self._edge_block(graph).edges,
                               [-1, self.n_heads, self.d_k])
            attention_graph = self._attention(node_values=values,
                                              node_keys=keys,
                                              node_queries=queries,
                                              edges=edges,
                                              attention_graph=graph)
        else:
            attention_graph = self._attention(node_values=values,
                                              node_keys=keys,
                                              node_queries=queries,
                                              attention_graph=graph)

        total_num_nodes = tf.reduce_sum(attention_graph.n_node)
        multihead_combined = tf.reshape(attention_graph.nodes,
                                        [total_num_nodes, self.d_model])

        attention_graph = attention_graph.replace(nodes=multihead_combined)
        output_graph = self._output_block(attention_graph)

        return output_graph


class SelfAttention(snt.AbstractModule):
    """Multi-head self-attention module.

    Modification of <venv>/lib/python3.6/site-packages/graph_nets/modules.py

    The module is based on the following three papers:
     * A simple neural network module for relational reasoning (RNs):
         https://arxiv.org/abs/1706.01427
     * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
     * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.

    The input to the modules consists of a graph containing values for each node
    and connectivity between them, a tensor containing keys for each node
    and a tensor containing queries for each node.

    The self-attention step consist of updating the node values, with each new
    node value computed in a two step process:
    - Computing the attention weights between each node and all of its senders
     nodes, by calculating sum(sender_key*receiver_query) and using the softmax
     operation on all attention weights for each node.
    - For each receiver node, compute the new node value as the weighted average
     of the values of the sender nodes, according to the attention weights.
    - Nodes with no received edges, get an updated value of 0.

    Values, keys and queries contain a "head" axis to compute independent
    self-attention for each of the heads.

    """

    def __init__(self, name="self_attention"):
        """Inits the module.

        Args:
            name: The module name.
        """
        super(SelfAttention, self).__init__(name=name)
        self._normalizer = modules._unsorted_segment_softmax

    def _build(self, node_values, node_keys, node_queries, edges, attention_graph):
        """Connects the multi-head self-attention module.

        The self-attention is only computed according to the connectivity of the
        input graphs, with receiver nodes attending to sender nodes.

        Args:
          node_values: Tensor containing the values associated to each of the nodes.
            The expected shape is [total_num_nodes, num_heads, key_size].
          node_keys: Tensor containing the key associated to each of the nodes. The
            expected shape is [total_num_nodes, num_heads, key_size].
          node_queries: Tensor containing the query associated to each of the nodes.
            The expected shape is [total_num_nodes, num_heads, query_size]. The
            query size must be equal to the key size.
          edges: Tensor [total_num_edges, num_heads, key size]
          attention_graph: Graph containing connectivity information between nodes
            via the senders and receivers fields. Node A will only attempt to attend
            to Node B if `attention_graph` contains an edge sent by Node A and
            received by Node B.

        Returns:
          An output `graphs.GraphsTuple` with updated nodes containing the
          aggregated attended value for each of the nodes with shape
          [total_num_nodes, num_heads, value_size].

        Raises:
          ValueError: if the input graph does not have edges.
        """
        # Sender nodes put their keys and values in the edges.
        # [total_num_edges, num_heads, query_size]
        sender_keys = blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_keys))
        # [total_num_edges, num_heads, value_size]
        sender_values = blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_values))

        # Receiver nodes put their queries in the edges.
        # [total_num_edges, num_heads, key_size]
        receiver_queries = blocks.broadcast_receiver_nodes_to_edges(
            attention_graph.replace(nodes=node_queries))

        # Attention weight for each edge.
        # [total_num_edges, num_heads]
        attention_weights_logits = tf.reduce_sum(
            sender_keys * receiver_queries, axis=-1)
        normalized_attention_weights = modules._received_edges_normalizer(
            attention_graph.replace(edges=attention_weights_logits),
            normalizer=self._normalizer)

        # This is the modification from the original
        sender_values += edges

        # Attending to sender values according to the weights.
        # [total_num_edges, num_heads, embedding_size]
        attented_edges = sender_values * normalized_attention_weights[..., None]

        # Summing all of the attended values from each node.
        # [total_num_nodes, num_heads, embedding_size]
        received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(
            reducer=tf.unsorted_segment_sum)
        aggregated_attended_values = received_edges_aggregator(
            attention_graph.replace(edges=attented_edges))

        return attention_graph.replace(nodes=aggregated_attended_values)
