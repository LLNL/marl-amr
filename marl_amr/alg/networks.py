from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

from functools import partial

import numpy as np
import sonnet as snt
import tensorflow as tf

from marl_amr.alg import custom_graphnet


def mlp(input_ph, n_outputs, config):
    """Fully-connected feedforward neural network.

    Args:
        input_ph: TF placeholder
        n_outputs: int
        config: ConfigDict object
    """
    list_hidden_size = config.list_hidden_size
    h = tf.layers.dense(inputs=input_ph, units=list_hidden_size[0],
                        activation=tf.nn.relu, use_bias=True, name='h1')
    for idx, size in enumerate(list_hidden_size[1:]):
        h = tf.layers.dense(inputs=h, units=size, activation=tf.nn.relu,
                            use_bias=True, name='h%d'%(idx+2))

    out = tf.layers.dense(inputs=h, units=n_outputs, activation=None,
                          use_bias=True, name='out')

    return out


def conv(t_input, scope, n_filters=6, k=(3, 3), s=(1, 1), data_format='NHWC'):
    """Convolutional layer.

    Args:
        t_input: TF placeholder
        scope: string for TF scope
        n_filter: number of convolutional filters
        k: 2-tuple, height and width of each filter
        s: 2-tuple, size of stride in each direction
        data_format: either 'NHWC' for channel-last or 'NCHW' for channel-first

    Returns: tf.nn.conv2d
    """
    if data_format == 'NHWC':
        channel_axis = 3
        strides = [1, s[0], s[1], 1]
        b_shape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_axis = 1
        strides = [1, 1, s[0], s[1]]
        b_shape = [1, n_filters, 1, 1]
    n_in = t_input.get_shape()[channel_axis].value
    w_shape = [k[0], k[1], n_in, n_filters]
    with tf.compat.v1.variable_scope(scope):
        w = tf.compat.v1.get_variable('w', w_shape)
        b = tf.compat.v1.get_variable('b', b_shape)
        out = b + tf.nn.conv2d(t_input, w, strides=strides, padding='SAME',
                               data_format=data_format)

    return out


def convnet(t_input, f=6, k=(3, 3), s=(1, 1), repeat=1):
    """Convolutional layer with flattening."""
    h = t_input
    for idx in range(1, repeat+1):
        h = tf.nn.relu(conv(h, 'c%d'%idx, f, k, s))
    size = np.prod(h.get_shape().as_list()[1:])
    conv_flat = tf.reshape(h, [-1, size])

    return conv_flat


def conv_mlp(input_ph, n_outputs, config):
    """Conv layer followed by MLP layers.

    Args:
        input_ph: TF placeholder
        n_outputs: integer
        config: ConfigDict object

    Returns: [batch, n_outputs]
    """    
    fc_input = convnet(input_ph, config.n_filters, config.kernel,
                       config.stride)
    out = mlp(fc_input, n_outputs, config)

    return out


def graph_local_q(input_graph, config, node_action_dim, dueling=False,
                  noisy=False, mode='train'):
    """A policy based on InteractionNetwork.

    See: https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/graph_nets_basics.ipynb
    for a tutorial.

    Maintains three distinct graph propagation layers.
    The first layer maps the high-dimensional image obs at each node
    to a lower dimensional vector for that node.
    The second layer performs r recurrent propagation steps, 
    with equal input and output sizes.
    The last layer outputs a vector of size <node_action_dim>  at each node.

    Args:
        input_graph: a GraphTuple object returned by utils_tf
        configs: ConfigDict
        node_action_dim: int, size of action space of each element
        dueling: Bool, whether to use Dueling Network
        noisy: Bool, whether to use NoisyLinear layer
        mode: str, 'train' or 'eval' used by noisy only

    Returns:
        tf.Tensor of node vectors with shape [batch, node_action_dim],
        list of custom_graphnet.NoisyLinear objects (empty if noisy=False)
    """
    if noisy:
        list_node_nl = []
        list_edge_nl = []
        for idx in range(3):
            if idx < 2:
                node_nl = custom_graphnet.NoisyLinear(config.output_node_size, mode)
                list_node_nl.append(node_nl)
                edge_nl = custom_graphnet.NoisyLinear(config.output_edge_size, mode)
                list_edge_nl.append(edge_nl)
            else:
                if dueling:
                    node_nl_v = custom_graphnet.NoisyLinear(1, mode)
                    list_node_nl.append(node_nl_v)
                    node_nl_a = custom_graphnet.NoisyLinear(node_action_dim, mode)
                    list_node_nl.append(node_nl_a)
                    edge_nl_v = custom_graphnet.NoisyLinear(config.output_edge_size, mode)
                    edge_nl_a = custom_graphnet.NoisyLinear(config.output_edge_size, mode)
                    list_edge_nl.append(edge_nl_v)
                    list_edge_nl.append(edge_nl_a)
                else:
                    node_nl = custom_graphnet.NoisyLinear(node_action_dim, mode)
                    list_node_nl.append(node_nl)
                    edge_nl = custom_graphnet.NoisyLinear(config.output_edge_size, mode)
                    list_edge_nl.append(edge_nl)

        mlp_node_layer = lambda: snt.Sequential([
            custom_graphnet.NoisyLinear(config.output_node_size, mode),
            tf.nn.relu])
        mlp_edge_layer = lambda: snt.Sequential([
            custom_graphnet.NoisyLinear(config.output_edge_size, mode),
            tf.nn.relu])
    else:
        mlp_node_layer = lambda: snt.Sequential([
            snt.Linear(output_size=config.output_node_size),
            tf.nn.relu])
        mlp_edge_layer = lambda: snt.Sequential([
            snt.Linear(output_size=config.output_edge_size),
            tf.nn.relu])

    # Need to use a GraphIndependent to operate on node attributes
    # independently from edges, to convert node attributes from
    # [batch, h,w,c] into [batch, one single dimension]
    # Otherwise InteractionNetwork can't concatenate node and edge attributes
    if 'conv' in config and config.conv:
        first_node_layer = lambda: snt.Sequential([
            snt.Conv2D(config.n_filters, config.kernel, config.stride,
                       name='first_node_layer_conv'),
            tf.nn.relu,
            snt.BatchFlatten(),
            snt.Linear(output_size=config.output_node_size,
                       name='first_node_layer_linear'),
            tf.nn.relu])
        graph_layer_0 = modules.GraphIndependent(
            node_model_fn = first_node_layer)

    def lambdify(layer, relu=True):
        if relu:
            return (lambda: snt.Sequential([layer, tf.nn.relu]))
        else:
            return (lambda: layer)

    graph_layer_1 = modules.InteractionNetwork(
        edge_model_fn = lambdify(list_edge_nl[0]) if noisy else mlp_edge_layer,
        node_model_fn = lambdify(list_node_nl[0]) if noisy else mlp_node_layer
    )

    graph_layer_2 = modules.InteractionNetwork(
        edge_model_fn = lambdify(list_edge_nl[1]) if noisy else mlp_edge_layer,
        node_model_fn = lambdify(list_node_nl[1]) if noisy else mlp_node_layer
    )

    if dueling:
        # Value
        graph_layer_3_v = modules.InteractionNetwork(
            edge_model_fn=lambdify(list_edge_nl[2]) if noisy else mlp_edge_layer,
            node_model_fn=(lambdify(list_node_nl[2], False) if noisy else
                           (lambda: snt.Linear(output_size=1)))
        )
        # Advantage
        graph_layer_3_a = modules.InteractionNetwork(
            edge_model_fn=lambdify(list_edge_nl[3]) if noisy else mlp_edge_layer,
            node_model_fn=(lambdify(list_node_nl[3], False) if noisy else
                           (lambda: snt.Linear(output_size=node_action_dim)))
        )
    else:
        graph_layer_3 = modules.InteractionNetwork(
            edge_model_fn=lambdify(list_edge_nl[2]) if noisy else mlp_edge_layer,
            node_model_fn=(lambdify(list_node_nl[2], False) if noisy else
                           (lambda: snt.Linear(output_size=node_action_dim)))
        )

    if 'conv' in config and config.conv:
        previous_graph = graph_layer_0(input_graph)
    else:
        previous_graph = input_graph
    previous_graph = graph_layer_1(input_graph)
    for _ in range(config.num_recurrent_passes):
        previous_graph = graph_layer_2(previous_graph)
    if dueling:
        value = graph_layer_3_v(previous_graph)
        advantage = graph_layer_3_a(previous_graph)
        output_nodes = (value.nodes + advantage.nodes - tf.reduce_sum(
            advantage.nodes, axis=1, keep_dims=True) / node_action_dim)
    else:
        output_graph = graph_layer_3(previous_graph)
        output_nodes = output_graph.nodes

    list_nl = (list_edge_nl + list_node_nl) if noisy else []

    return tf.cast(output_nodes, tf.float32), list_nl


def graph_attention_q(input_graph, config, node_action_dim, dueling=False):
    """Defines one forward pass through attention-based graph network.

    Args:
        input_graph: a GraphTuple object returned by utils_tf
        config: ConfigDict
        node_action_dim: int, size of action space of each element
        dueling: Bool, whether to use Dueling Network

    Returns:
        tf.Tensor of node vectors with shape [batch, node_action_dim],
    """
    mlp_node_layer = lambda: snt.Sequential([
        snt.Linear(output_size=config.d_model),
        tf.nn.relu])
    att_proj_layer = lambda: snt.Linear(output_size=config.d_model)

    # -------------------- Define graph layers -------------------- #
    input_layer = modules.GraphIndependent(
        edge_model_fn = ((lambda: snt.Linear(output_size=config.d_model)) if
                         config.att_use_edges else None),
        node_model_fn = lambda: snt.Linear(output_size=config.d_model)
    )
    layer_norm = modules.GraphIndependent(
        edge_model_fn = ((lambda: snt.LayerNorm(axis=-1)) if config.att_use_edges
                         else None),
        node_model_fn = lambda: snt.LayerNorm(axis=-1)
    )
    list_att_layers = []
    list_ff_layers = []
    for _ in range(config.num_att_layers):
        list_att_layers.append(
            custom_graphnet.CustomGraphAttentionNetwork(
                node_model_fn=att_proj_layer,
                output_model_fn=att_proj_layer,
                use_edges=config.att_use_edges,
                d_model=config.d_model,
                n_heads=config.n_heads
            )
        )
        list_ff_layers.append(
            modules.GraphIndependent(
                edge_model_fn = mlp_node_layer if config.att_use_edges else None,
                node_model_fn = mlp_node_layer
            )
        )

    if dueling:
        if config.output_independent:
            out_layer_v = modules.GraphIndependent(
                edge_model_fn = None,
                node_model_fn = lambda: snt.Linear(output_size=1)
            )
            out_layer_a = modules.GraphIndependent(
                edge_model_fn = None,
                node_model_fn = lambda: snt.Linear(output_size=node_action_dim)
            )
        else:
            # Value
            out_layer_v = custom_graphnet.CustomGraphAttentionNetwork(
                node_model_fn=att_proj_layer,
                output_model_fn=lambda: snt.Linear(output_size=1),
                use_edges=config.att_use_edges,
                d_model=config.d_model,
                n_heads=config.n_heads
            )
            # Advantage
            out_layer_a = custom_graphnet.CustomGraphAttentionNetwork(
                node_model_fn=att_proj_layer,
                output_model_fn=lambda: snt.Linear(output_size=node_action_dim),
                use_edges=config.att_use_edges,
                d_model=config.d_model,
                n_heads=config.n_heads
            )
    else:
        if config.output_independent:
            out_layer = modules.GraphIndependent(
                edge_model_fn = None,
                node_model_fn = lambda: snt.Linear(output_size=node_action_dim)
            )
        else:
            out_layer = custom_graphnet.CustomGraphAttentionNetwork(
                node_model_fn=att_proj_layer,
                output_model_fn=lambda: snt.Linear(output_size=node_action_dim),
                use_edges=config.att_use_edges,
                d_model=config.d_model,
                n_heads=config.n_heads
            )

    # -------------------- Forward pass -------------------- #
    previous_graph = input_layer(input_graph)
    # Each recurrent pass uses the same layer(s)
    for _ in range(config.num_recurrent_passes):
        # Within each pass, can have multiple non-shared attention layers
        for i in range(config.num_att_layers):
            # run attention layer
            edges = previous_graph.edges
            nodes = previous_graph.nodes
            previous_graph = list_att_layers[i](previous_graph)
            if config.residual:
                previous_graph = previous_graph.replace(
                    edges=previous_graph.edges + edges,
                    nodes=previous_graph.nodes + nodes)
            if config.layer_norm:
                previous_graph = layer_norm(previous_graph)

            # run feedforward layer
            edges = previous_graph.edges
            nodes = previous_graph.nodes
            previous_graph = list_ff_layers[i](previous_graph)
            if config.residual:
                previous_graph = previous_graph.replace(
                    edges=previous_graph.edges + edges,
                    nodes=previous_graph.nodes + nodes)
            if config.layer_norm:
                previous_graph = layer_norm(previous_graph)

    # run output layer
    if dueling:
        value = out_layer_v(previous_graph)
        advantage = out_layer_a(previous_graph)
        output_nodes = (value.nodes + advantage.nodes - tf.reduce_sum(
            advantage.nodes, axis=1, keep_dims=True) / node_action_dim)
    else:
        output_graph = out_layer(previous_graph)
        output_nodes = output_graph.nodes

    return tf.cast(output_nodes, tf.float32)


def run_hypernet(input_graph, layer, final_layer, num_passes,
                 e_size, n_size, name):
    """Runs hypernet.

    Args:
        input_graph: graph_nets.GraphsTuple
        layer: graph_nets.modules.<architecture>
        final_layer: graph_nets.modules.<architecture>
        num_passes: int, number of recurrent message-passes
        e_size: int, size of subcomponent of output used for mixer edge
        n_size: int, size of subcomponent of output used for mixer node

    Returns:
        tuple of two tf.Tensors, interpreted as edge and node weights or bias
            used in the subsequent mixer graphnet.
    """
    hypernet_graph = input_graph
    if num_passes == 1:
        hypernet_graph = final_layer(hypernet_graph)
    else:
        for _ in range(num_passes - 1):
            hypernet_graph = layer(hypernet_graph)
        hypernet_graph = final_layer(hypernet_graph)

    # At this point, hypernet_graph is a batch of graphs, graph i has k_i nodes.
    # For each graph, set global to be element-wise sum of node features
    reducer = tf.unsorted_segment_sum
    hypernet_nodes_to_globals = hypernet_graph.replace(
        globals=blocks.NodesToGlobalsAggregator(reducer, name)(hypernet_graph))
    # [batch, hyper_<1>_<2>_output_size], where <1> is layer number of hypernet
    # and <2> is 'w' (weight) or 'b' (bias)
    # hypernet_output = tf.squeeze(hypernet_nodes_to_globals.globals)
    hypernet_output = hypernet_nodes_to_globals.globals

    # split hypernet output into parts for edge and node
    # blocks of mixer InteractionNetwork.
    # [batch, hyper_<1 or 2>_<e or n>_output_size]
    mixer_edge_params, mixer_node_params = tf.split(
        hypernet_output, num_or_size_splits=[e_size, n_size], axis=1)

    return mixer_edge_params, mixer_node_params


def hyper_graphnet_mixer(obs_graph, q_i_graph, config, dim_edge):
    """Mixing network with weights generated by a hypernetwork.

    Both the hypernetwork and mixing network are graphnets.
    Mixer is monotonically increasing in each q_i because hypernetwork generates
    non-negative weights.

    Block diagram:
    obs_graph --> |hyper-graphnet| --> weights for mixer graphnet
                                               |
                          ----------------------
                          |
                          V
    q_i_graph --> |graphnet mixer| --> Q(s,a)

    Each batch entry of hypernet's output should be used as the mixer
    weight for the corresponding batch entry of q_i_graph.
    Vectorizing this is done by the use of custom_graphnet.CustomLinear
    (and CustomNodeBlock and CustomEdgeBlock),
    which repeats hypernet output[i] by (n_i) times, where (n_i)
    is the number of nodes or edges inside q_i_graph[i].

    Args:
        obs_graph: a GraphTuple object returned by utils_tf,
                   each node is an agent's individual observation.
        q_i_graph: a GraphTuple object returned by utils_tf,
                   each node is the Q^i(o^i,a^i) value of an agent.
        config: ConfigDict
        dim_edge: int, initial edge attribute dimension

    Returns:
        tf.Tensor of shape [batch]
    """
    # ------------------------ define layer sizes ------------------------- #
    # input dims in the first message-passing round of mixer
    # 2 is for the q-values at connected nodes
    mixer_1_e_input_size = 2 + dim_edge
    # 1 is for the q-value at node, plus the size of updated edge
    mixer_1_n_input_size = 1 + config.mixer_1_e_output_size

    # Weight output of hypernetwork_1 will be split into two parts,
    # each of which is reshaped into a matrix with the following sizes.
    hyper_1_e_output_size = (mixer_1_e_input_size *
                             config.mixer_1_e_output_size)
    hyper_1_n_output_size = (mixer_1_n_input_size *
                             config.mixer_1_n_output_size)
    hyper_1_w_output_size = hyper_1_e_output_size + hyper_1_n_output_size
    # Bias output of hypernetwork_1 will be split into two parts,
    # each of which is a vector
    hyper_1_b_output_size = (config.mixer_1_e_output_size +
                             config.mixer_1_n_output_size)

    # Weight output of hypernetwork_2 will be split into two parts
    # They are used in mixer layer 2
    # Mixer layer 2's input graph is mixer layer 1's output graph
    # Mixer edge layer input is [e_k, v_{r_k}, v_{s_k}]
    mixer_2_e_input_size = config.mixer_1_e_output_size + 2 * (
        config.mixer_1_n_output_size)
    hyper_2_e_output_size = (mixer_2_e_input_size *
                             config.mixer_2_e_output_size)
    # Mixer node layer input is [agg e'_i, v_i]
    mixer_2_n_input_size = (config.mixer_2_e_output_size +
                            config.mixer_1_n_output_size)
    # Final output node size is 1, written out for clarity.
    hyper_2_n_output_size = (mixer_2_n_input_size * 1)
    hyper_2_w_output_size = hyper_2_e_output_size + hyper_2_n_output_size
    # Bias output of of hypernetwork_2 will be split into two parts
    hyper_2_b_output_size = (config.mixer_2_e_output_size + 1)
    # ---------------------------------------------------------------------- #

    # -------------------- hyper-graphnet layers -------------------- #
    hypernet_edge_layer = lambda: snt.Linear(output_size=config.hyper_edge_size)
    hypernet_node_layer = lambda: snt.Linear(output_size=config.hyper_node_size)

    # 1a. Hypernet layers that generate weights for mixer layer 1
    hypernet_1_w_layer = modules.InteractionNetwork(
        edge_model_fn = hypernet_edge_layer,
        node_model_fn = hypernet_node_layer,
        name = 'hypernet_1_w_layer'
    )
    # Only final node layer needs to produce non-negative values:
    # it is applied after edge layer,
    # and Hypernet output will be taken from nodes
    hypernet_1_node_layer_abs = lambda: snt.Sequential([
        snt.Linear(output_size=hyper_1_w_output_size),
        tf.math.abs])
    hypernet_1_w_layer_final = modules.InteractionNetwork(
        edge_model_fn = hypernet_edge_layer,
        node_model_fn = hypernet_1_node_layer_abs,
        name = 'hypernet_1_w_layer_final'
    )

    # 1b. Hypernet layers that generate bias for mixer layer 1
    hypernet_1_b_layer = modules.InteractionNetwork(
        edge_model_fn = hypernet_edge_layer,
        node_model_fn = hypernet_node_layer,
        name = 'hypernet_1_b_layer'
    )
    hypernet_1_b_layer_final = modules.InteractionNetwork(
        edge_model_fn = hypernet_edge_layer,
        node_model_fn = lambda: snt.Linear(output_size=hyper_1_b_output_size),
        name = 'hypernet_1_b_layer_final'
    )

    # 2a. Hypernet layers that generate weights for mixer layer 2
    hypernet_2_w_layer = modules.InteractionNetwork(
        hypernet_edge_layer, hypernet_node_layer, name='hypernet_2_w_layer')
    hypernet_2_node_layer_abs = lambda: snt.Sequential([
        snt.Linear(output_size=hyper_2_w_output_size),
        tf.math.abs])
    hypernet_2_w_layer_final = modules.InteractionNetwork(
        hypernet_edge_layer, hypernet_2_node_layer_abs,
        name='hypernet_2_w_layer_final')

    # 2b. Hypernet layers that generate bias for mixer layer 2
    hypernet_2_b_layer = modules.InteractionNetwork(
        hypernet_edge_layer, hypernet_node_layer, name='hypernet_2_b_layer')
    hypernet_2_b_layer_final = modules.InteractionNetwork(
        edge_model_fn = hypernet_edge_layer,
        node_model_fn = lambda: snt.Linear(output_size=hyper_2_b_output_size),
        name = 'hypernet_2_b_layer_final'
    )
    # ---------------------------------------------------------------------- #

    # -------------------- hyper-graphnet message passing --------------------#
    mixer_1_edge_weights, mixer_1_node_weights = run_hypernet(
        obs_graph, hypernet_1_w_layer, hypernet_1_w_layer_final,
        config.hypernet_num_message_passes,
        hyper_1_e_output_size, hyper_1_n_output_size, 'run_hypernet_1_w')

    mixer_1_edge_bias, mixer_1_node_bias = run_hypernet(
        obs_graph, hypernet_1_b_layer, hypernet_1_b_layer_final,
        config.hypernet_num_message_passes,
        config.mixer_1_e_output_size, config.mixer_1_n_output_size,
        'run_hypernet_1_b')

    mixer_2_edge_weights, mixer_2_node_weights = run_hypernet(
        obs_graph, hypernet_2_w_layer, hypernet_2_w_layer_final,
        config.hypernet_num_message_passes,
        hyper_2_e_output_size, hyper_2_n_output_size, 'run_hypernet_2_w')

    mixer_2_edge_bias, mixer_2_node_bias = run_hypernet(
        obs_graph, hypernet_2_b_layer, hypernet_2_b_layer_final,
        config.hypernet_num_message_passes,
        config.mixer_2_e_output_size, 1, 'run_hypernet_2_b')
    # ---------------------------------------------------------------------- #

    # --------------- reshape weights into batch of matrices --------------- #
    batch_size = tf.shape(mixer_1_node_weights)[0]
    # 1. mixer layer 1
    # [batch, mixer e/n input dim, mixer e/n output dim]
    mixer_1_edge_weights = tf.reshape(mixer_1_edge_weights,
                                      [batch_size, mixer_1_e_input_size,
                                      config.mixer_1_e_output_size])
    mixer_1_edge_bias = tf.reshape(mixer_1_edge_bias,
                                   [batch_size, config.mixer_1_e_output_size])
    mixer_1_node_weights = tf.reshape(mixer_1_node_weights,
                                      [batch_size, mixer_1_n_input_size,
                                      config.mixer_1_n_output_size])
    mixer_1_node_bias = tf.reshape(mixer_1_node_bias,
                                   [batch_size, config.mixer_1_n_output_size])

    # 2. mixer layer 2
    mixer_2_edge_weights = tf.reshape(
        mixer_2_edge_weights, [batch_size, mixer_2_e_input_size,
                               config.mixer_2_e_output_size])
    mixer_2_edge_bias = tf.reshape(mixer_2_edge_bias,
                                   [batch_size, config.mixer_2_e_output_size])
    mixer_2_node_weights = tf.reshape(
        mixer_2_node_weights, [batch_size, mixer_2_n_input_size, 1])
    mixer_2_node_bias = tf.reshape(mixer_2_node_bias, [batch_size, 1])
    # ---------------------------------------------------------------------- #

    # ------------- graphnet mixer uses hypernet output as weights ----------- #
    mixer_1_edge_layer = lambda: snt.Sequential([
        snt.Module(name='mixer_1_edge_matmul',
                   build=partial(custom_graphnet.CustomLinear,
                                 w=mixer_1_edge_weights, b=mixer_1_edge_bias)),
        tf.nn.elu])
    mixer_1_node_layer = lambda: snt.Sequential([
        snt.Module(name='mixer_1_node_matmul',
                   build=partial(custom_graphnet.CustomLinear,
                                 w=mixer_1_node_weights, b=mixer_1_node_bias)),
        tf.nn.elu])
    mixer_layer_1 = custom_graphnet.CustomInteractionNetwork(
        edge_model_fn = mixer_1_edge_layer,
        node_model_fn = mixer_1_node_layer,
        name = 'mixer_layer_1'
    )

    mixer_edge_final_layer = lambda: snt.Module(
        name='mixer_final_edge_matmul',
        build=partial(custom_graphnet.CustomLinear,
                      w=mixer_2_edge_weights, b=mixer_2_edge_bias))
    mixer_node_final_layer = lambda: snt.Module(
        name='mixer_final_node_matmul',
        build=partial(custom_graphnet.CustomLinear,
                      w=mixer_2_node_weights, b=mixer_2_node_bias))
    mixer_final_layer = custom_graphnet.CustomInteractionNetwork(
        edge_model_fn = mixer_edge_final_layer,
        node_model_fn = mixer_node_final_layer,
        name = 'mixer_final_layer'
    )

    # ------------------------ mixer message passing ------------------------ #

    mixer_graph = mixer_layer_1(q_i_graph)
    mixer_graph = mixer_final_layer(mixer_graph)

    reducer = tf.unsorted_segment_sum
    mixer_nodes_to_globals = mixer_graph.replace(
        globals=blocks.NodesToGlobalsAggregator(reducer, 'n_to_g')(mixer_graph))
    # [batch]
    q_tot = tf.reshape(mixer_nodes_to_globals.globals, [-1])

    return q_tot
