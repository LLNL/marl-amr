"""Evaluates custom refinement strategies."""

import commentjson
import os
import time

from marl_amr.envs import util
from marl_amr.tools import amr_utils
from marl_amr.tools import estimators
from marl_amr.tools import policies

from copy import deepcopy
from multiprocessing import Process

import mfem.ser as mfem
import numpy as np


class DummyEnv(object):
    def __init__(self, solver):
        self.solver = solver


def eval_custom_fixed_mesh(option, config_env, n_episodes, n_steps,
                           name='', save_mesh=False, write_csv=False,
                           save_dir=''):
    """Measures cost (DoF) and error of a fixed mesh.

    h-refinement only. Hard-coded to 2 steps

    Args:
        option: 'coarse' or 'fine'
        config_env: dict
        n_episodes: int
        n_steps: int
        name: str
        save_mesh: bool

    Returns:
        error: float
        sum_of_dofs: int
    """
    if write_csv:
        f_name = '_'.join([name, option]) + '.csv'
        path = os.path.join(save_dir, f_name)
        with open(path, 'w') as f:
            f.write('episode,init_global_error,episode_steps,sum_of_dofs,'
                    'true_global_error\n')

    solver = util.GetSolver(config_env['solver_name'],
                            **config_env['solver'])
    solver.error_threshold = config_env['error_threshold']
    solver.SetFinalTime(config_env['t_final'])

    for idx_episode in range(1, n_episodes+1):

        solver.Reset()
        init_global_error = solver.GetGlobalError()
        get_fixed_mesh(solver, option, config_env['max_depth'])
        sum_of_dofs = solver.fespace.GetTrueVSize()
        error = solver.GetGlobalError()

        # Run steps
        t = 0
        if save_mesh:
            amr_utils.output_mesh(DummyEnv(solver), os.getcwd(), t)
        while t < n_steps:
            solver.Step()
            t += 1
            sum_of_dofs += solver.fespace.GetTrueVSize()
            error = solver.GetGlobalError()
            if save_mesh:
                amr_utils.output_mesh(DummyEnv(solver), os.getcwd(), t)

        error = solver.GetGlobalError()

        if write_csv:
            with open(path, 'a') as f:
                f.write('%d,%.6e,%d,%d,%.6e\n' % (
                    idx_episode, init_global_error, t, sum_of_dofs, error))
        else:
            print('error', error)
            print('sum_of_dofs', sum_of_dofs)

    return error, sum_of_dofs


def custom_Reset(solver, option, max_depth):
    """Modifed version of AdvectionSolver.Reset()

    Args:
        solver: instance of advection.solvers.?
        option: str, 'fine' or 'coarse'
        max_depth: int, max h-ref depth
    """
    solver.t = 0.0

    if solver.solver_initialized: solver.Delete()
    solver.SetupMesh()
    solver.SetupCoefficients()
    solver.SetupFEM()
    solver.SetupODESolver()

    solver.solver_initialized = True

    get_fixed_mesh(solver, option, max_depth)
    solver.SetupEstimator()


def get_fixed_mesh(solver, option, max_depth):
    """Modifed version of AdvectionSolver.SetInitialCondition()

    If option == 'fine', then refine everything to max_depth

    Args:
        solver: instance of advection.solvers.?
        option: str, 'fine' or 'coarse'
        max_depth: int, max h-ref depth
    """
    solver.solution = mfem.GridFunction(solver.fespace);
    solver.solution.ProjectCoefficient(solver.true_solution)

    if option == 'fine':
        for _ in range(max_depth):
            elems = list(range(solver.mesh.GetNE()))
            element_action_list = [1] * solver.mesh.GetNE()
            solver.hRefine(element_action_list, solver.aniso, max_depth)

    solver.solution.ProjectCoefficient(solver.true_solution)
    

def eval_refine_all(config_env, n_steps):
    """Evalutes the policy that refines everything at each step."""

    solver = util.GetSolver(config_env['solver_name'], **config_env['solver'])
    # solver.SetRefinementMode(config_env['refinement_mode'])
    solver.error_threshold = config_env['error_threshold']
    solver.SetFinalTime(config_env['t_final'])
    solver.Reset()
    sum_of_dofs = solver.fespace.GetTrueVSize()

    # Run steps
    t = 0
    amr_utils.output_mesh(DummyEnv(solver), os.getcwd(), t)
    while t < n_steps:
        element_action_list = [1] * solver.mesh.GetNE()
        solver.hRefine(element_action_list, solver.aniso, config_env['max_depth'])
        solver.Step()
        t += 1
        sum_of_dofs += solver.fespace.GetTrueVSize()
        amr_utils.output_mesh(DummyEnv(solver), os.getcwd(), t)

    error = solver.GetGlobalError()

    print('error', error)
    print('sum_of_dofs', sum_of_dofs)

    return error, sum_of_dofs


def eval_double_threshold(config_env,
                          array_low=[], array_high=[],
                          n_episodes=1, n_steps=0,
                          fname='temp.csv', exp_name='adv',
                          save_error_vs_time=False, save_mesh=False,
                          seed=12340, save_dir=''):
    """Evaluates policy that refines (derefines) elems above (below) threshold.

    Saves results to CSV files inside this directory.

    Args:
        config_env: dict
        low_range: 2-tuple
        high_range: 2-tuple
    """
    if n_episodes == 1:
        with open(os.path.join(save_dir, 'pareto.csv'), 'w') as f:
            f.write('method,low,high,true_global_error,sum_of_dofs\n')

    if save_error_vs_time:
        with open(os.path.join(save_dir, fname), 'w') as f:
            f.write('time,global_error,sum_of_dofs\n')

    count = 0
    for low in array_low:
        for high in array_high:

            np.random.seed(seed)
            solver = util.GetSolver(config_env['solver_name'],
                                    **config_env['solver'])
            # solver.SetRefinementMode(config_env['refinement_mode'])
            solver.error_threshold = config_env['error_threshold']
            solver.SetFinalTime(config_env['t_final'])

            est = estimators.TrueErrorEstimator(solver)

            policy = policies.DoubleThresholdPolicy(est, low, high)
            policy.SetActionMapping(action_type='increment_current')

            if n_episodes > 1:
                f_name_thres = '_'.join([exp_name, str(high)]) + '.csv'
                with open(os.path.join(save_dir, f_name_thres), 'w') as f:
                    f.write('episode,init_global_error,episode_steps,'
                            'sum_of_dofs,true_global_error\n')

            t_act = 0
            t_env = 0
            total_steps = 0
            for idx_episode in range(1, n_episodes + 1):
                solver.Reset()
                init_global_error = solver.GetGlobalError()
                sum_of_dofs = solver.fespace.GetTrueVSize()

                error = solver.GetGlobalError()[0]
                if save_error_vs_time:
                    with open(os.path.join(save_dir, fname), 'a') as f:
                        f.write('{:f},{:e},{:d}\n'.format(0, error, sum_of_dofs))

                step = 0
                list_global_error = []
                list_sum_of_dofs = []
                if save_mesh:
                    amr_utils.output_mesh(
                        DummyEnv(solver), os.getcwd(), step, False)
                while step < n_steps:
                    t_start = time.time()
                    scores = est.ComputeScores()
                    ref = policy.ComputeRefinementActions()
                    deref = policy.ComputeDerefinementActions()
                    element_action_list = [0] * solver.mesh.GetNE()
                    for elem in range(len(scores)):
                        ref_val = ref[str(elem)]
                        deref_val = deref[str(elem)]
                        if ref_val == 0 and deref_val == 0:
                            element_action_list[elem] = 0
                        else:
                            element_action_list[elem] = (ref_val if ref_val == 1
                                                         else deref_val)
                    t_act += time.time() - t_start
                    total_steps += 1

                    step += 1
                    t_env_start = time.time()
                    solver.hRefine(element_action_list,
                                  aniso=config_env['solver']['aniso'],
                                  depth_limit=config_env['max_depth'])
                    if save_mesh:
                        amr_utils.output_mesh(DummyEnv(solver), os.getcwd(),
                                              step, False)

                    solver.Step()

                    sum_of_dofs += solver.fespace.GetTrueVSize()
                    error = solver.GetGlobalError()[0]
                    t_env += time.time() - t_env_start

                    if save_error_vs_time:
                        with open(os.path.join(save_dir, fname), 'a') as f:
                            f.write('{:f},{:e},{:d}\n'.format(
                                step*solver.t_step, error, sum_of_dofs))

                if n_episodes > 1:
                    with open(os.path.join(save_dir, f_name_thres), 'a') as f:
                        f.write('%d,%.6e,%d,%d,%.6e\n' % (
                            idx_episode, init_global_error, step, sum_of_dofs,
                            error))

                if save_mesh:
                    amr_utils.output_mesh(
                        DummyEnv(solver), os.getcwd(), step+1, False)
                error = solver.GetGlobalError()[0]

            if n_episodes == 1:
                with open(os.path.join(save_dir, 'pareto.csv'), 'a') as f:
                    f.write('{:.0e},{:e},{:e},{:e},{:d}\n'.format(
                        high, low, high, error, sum_of_dofs))
            else:
                with open(os.path.join(save_dir, f_name_thres), 'a') as f:
                    f.write('avg_t_action,%.6e\n' % (t_act/total_steps))
                    f.write('avg_t_env,%.6e\n' % (t_env/total_steps))
                    f.write('avg_t_episode,%.6e\n' % ((t_act + t_env)/n_episodes))
            count += 1

    return error, sum_of_dofs


def evaluate_global_error_vs_time(config_env, thres_low, thres_high,
                                  list_dt_multiplier, save_mesh=False,
                                  save_dir=''):
    """Runs threshold policy with various tstep.

    Args:
        config_env: dict
        thres_low: float
        thres_high: float
        list_dt_multiplier: list of ints
        save_mesh: Bool
    """
    # duplicate
    array_low = [thres_low]
    array_high = [thres_high]

    dt = config_env['solver']['dt']
    t_final = config_env['t_final']

    # go through each dt multiplier
    for mult in list_dt_multiplier:

        config_copy = deepcopy(config_env)
        t_step = mult * dt
        config_copy['solver']['t_step'] = t_step
        n_steps = t_final / t_step
        print('n_steps', n_steps)
        fname = 'true_error_t_step_{}_dt.csv'.format(mult)
        eval_double_threshold(config_copy, array_low, array_high,
                              n_episodes=1,
                              n_steps=n_steps, fname=fname,
                              save_error_vs_time=True,
                              save_mesh=save_mesh, save_dir=save_dir)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='name of config file in marl_amr.envs.configs')
    parser.add_argument('option', type=str,
                        choices=['coarse', 'fine', 'refine_all', 'dt',
                                 'err_vs_time'],
                        help='choice of function to run')
    parser.add_argument('--low_array', type=str, default='',
                        help='list of lower thresholds')
    parser.add_argument('--high_array', type=str, default='',
                        help='list of upper thresholds')
    parser.add_argument('--use_range', action='store_true')
    parser.add_argument('--low_range', type=str, default='1e-6,1e-4',
                        help='range in which to choose lower threshold')
    parser.add_argument('--high_range', type=str, default='5e-4,1e-2',
                        help='range in which to choose upper threshold')
    parser.add_argument('--n_episodes', type=int, default=1,
                        help='number of test episodes')
    parser.add_argument('--n_inc', type=int, default=1,
                        help='number of equally-spaced points between low_range'
                        ' and high_range')
    parser.add_argument('--name', type=str, default='',
                        help='name of test setup, do not include extension')
    parser.add_argument('--thres_low', type=float, default=1e-4,
                        help='derefines if error < thres_low')
    parser.add_argument('--thres_high', type=float, default=5e-4,
                        help='refines if error > thres_high')
    parser.add_argument('--multipliers', type=str, default='250',
                        help='comma-separated integers')
    parser.add_argument('--seed', type=int, default=12340)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('--multiprocess', action='store_true')
    args = parser.parse_args()

    with open('../envs/configs/{}.json'.format(args.config), 'r') as f:
        config_env = commentjson.load(f)['env']

    np.random.seed(args.seed)

    n_steps = int(config_env['t_final'] / config_env['solver']['t_step'])

    if args.save_dir != '':
        os.makedirs(args.save_dir, exist_ok=True)

    if args.option == 'refine_all':
        eval_refine_all(config_env, n_steps)
    elif args.option == 'dt':
        if args.use_range:
            low_range = list(map(float, args.low_range.split(',')))
            low_array = np.linspace(low_range[0], low_range[1], args.n_inc)
            high_range = list(map(float, args.high_range.split(',')))
            high_array = np.linspace(high_range[0], high_range[1], args.n_inc)
        else:
            low_array = list(map(float, args.low_array.split(',')))
            high_array = list(map(float, args.high_array.split(',')))
        if args.multiprocess:
            processes = []
            # assume only sweep high thres
            for idx_run in range(len(high_array)):
                high = [high_array[idx_run]]
                p = Process(target=eval_double_threshold, args=(
                    config_env, low_array, high,
                    args.n_episodes, n_steps, '', args.name,
                    False, args.save_mesh, args.seed, args.save_dir))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            eval_double_threshold(config_env, low_array, high_array,
                                  args.n_episodes, n_steps, exp_name=args.name,
                                  save_mesh=args.save_mesh, seed=args.seed,
                                  save_dir=args.save_dir)
    elif args.option == 'err_vs_time':
        list_dt_multiplier = list(map(int, args.multipliers.split(',')))
        evaluate_global_error_vs_time(config_env, args.thres_low,
                                      args.thres_high, list_dt_multiplier,
                                      args.save_mesh, args.save_dir)
    else:
        eval_custom_fixed_mesh(args.option, config_env, args.n_episodes, n_steps,
                               args.name, args.save_mesh, args.write_csv,
                               args.save_dir)
