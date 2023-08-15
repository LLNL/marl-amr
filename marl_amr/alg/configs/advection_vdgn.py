from marl_amr.alg.utils.configdict import ConfigDict
import numpy as np


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.batch_size = 16
    config.alg.beta_init = 0.01
    config.alg.buffer_size = 10000
    config.alg.ddqn = True
    config.alg.dueling = True
    config.alg.epsilon_div = 150000
    config.alg.epsilon_end = 0.01
    config.alg.epsilon_start = 0.5
    config.alg.explore_type = 'independent' # 'global' or 'independent'
    config.alg.gamma = 0.99
    config.alg.lr = 5e-4
    config.alg.multi_step = False
    config.alg.n_episodes = 400000
    config.alg.n_eval = 1
    config.alg.n_objectives = 1
    config.alg.n_test_episodes = 1
    config.alg.n_weights = 10
    config.alg.name = 'vdgn'
    config.alg.noisy_net = False
    config.alg.period = 100
    config.alg.prioritized_replay = True
    config.alg.priority_exponent = 0.5
    config.alg.priority_importance_exponent_start = 0.4
    config.alg.priority_normalize_weights = True
    config.alg.steps_per_train = 4
    config.alg.tau = 0.0112
    config.alg.uniform_sample_probability = 1e-3

    config.env = ConfigDict()
    config.env.agent_manager_use_tree = False
    config.env.agent_obs_type = 'self'
    config.env.debug = False
    config.env.dimensionless = True
    config.env.dof_threshold = 5620
    config.env.edge_feature_is_relative = True
    config.env.enable_deref = True
    config.env.error_threshold = 5.0e-4
    config.env.log_obs = True
    config.env.max_depth = 1
    config.env.multi_objective = False
    config.env.obs_uses_true_error = True
    config.env.observation_type = 'graph'
    config.env.observe_depth = True
    config.env.observe_dof_and_time_balance = False
    config.env.penalize_dof_excess = True
    config.env.reward_type = 'global'
    config.env.reward_uses_true_error = True
    config.env.solver = ConfigDict()
    config.env.solver.CFL = None
    config.env.solver.aniso = False
    config.env.solver.dt = 0.002
    config.env.solver.element_shape = 'quad'
    config.env.solver.error_method = 'projected'
    config.env.solver.initial_condition = ConfigDict()
    config.env.solver.initial_condition.coefficient = 'Gaussian2DCoefficient'
    # config.env.solver.initial_condition.coefficient = 'GaussianCoefficient'
    config.env.solver.initial_condition.params = ConfigDict()
    config.env.solver.initial_condition.params.randomize = 'uniform'
    config.env.solver.initial_condition.params.theta_range = [0.0, 1.0]
    config.env.solver.initial_condition.params.u0_range = [0.0, np.sqrt(2*1.5**2)]
    config.env.solver.initial_condition.params.w_range = [100, 100]
    config.env.solver.initial_condition.params.x0_range = [0.5, 1.5]
    config.env.solver.initial_condition.params.y0_range = [0.5, 1.5]
    # config.env.solver.initial_condition.params.randomize = 'discrete'
    # config.env.solver.initial_condition.params.theta_discrete = [0.125]
    # config.env.solver.initial_condition.params.u0_discrete = [np.sqrt(2*1.5**2)]
    # config.env.solver.initial_condition.params.w_discrete= [100]
    # config.env.solver.initial_condition.params.x0_discrete = [0.5]
    # config.env.solver.initial_condition.params.y0_discrete = [0.5]
    config.env.solver.initial_condition.params_test = ConfigDict()
    config.env.solver.initial_condition.params_test.randomize = 'discrete'
    config.env.solver.initial_condition.params_test.theta_discrete = [0.125]
    config.env.solver.initial_condition.params_test.u0_discrete = [1.5]
    config.env.solver.initial_condition.params_test.w_discrete= [100]
    config.env.solver.initial_condition.params_test.x0_discrete = [0.5]
    config.env.solver.initial_condition.params_test.y0_discrete = [0.5]
    config.env.solver.initial_condition.test_is_different = True
    config.env.solver.jit = True
    config.env.solver.nx = 16
    config.env.solver.ny = 16
    config.env.solver.order = 1
    config.env.solver.periodic = True
    config.env.solver.refine_IC = True
    config.env.solver.refinement_mode = 'h'
    config.env.solver.single_step = False
    config.env.solver.sx = 2.0
    config.env.solver.sy = 2.0
    config.env.solver.t_step = 0.25
    config.env.solver_name = 'advection'
    config.env.stopping_criteria = 'budget_or_time'
    config.env.t_final = 0.75
    config.env.t_history = 1

    config.main = ConfigDict()
    config.main.dir_name = 'nx16_ny16_depth1_tstep0p25_vdgn'
    config.main.dir_restore = ''
    config.main.exp_name = 'advection'
    config.main.gpu_id = 0
    config.main.gpu_min_memory = 512
    config.main.max_to_keep = 2
    config.main.model_name = 'model.ckpt'
    config.main.model_name_restore = ''
    config.main.n_parallel = 16
    config.main.n_train_parallel = 3
    config.main.resume_episode = 0
    config.main.resume_from_pretrained = False
    config.main.resume_training = False
    config.main.save_period = 100
    config.main.save_threshold = 3.2
    config.main.seed = 12343
    config.main.train_path = '../results/'
    # set to True to train on GPUs. Also see @ray.remote in runners.py
    config.main.use_gpu = False 

    config.nn = ConfigDict()
    config.nn.attention = True
    config.nn.att_use_edges = True
    config.nn.conv = False
    config.nn.d_model = 64
    config.nn.layer_norm = True
    config.nn.n_heads = 2
    config.nn.num_att_layers = 2
    config.nn.num_recurrent_passes = 3
    config.nn.output_edge_size = 32
    config.nn.output_node_size = 32
    config.nn.output_independent = False
    config.nn.residual = True

    return config
