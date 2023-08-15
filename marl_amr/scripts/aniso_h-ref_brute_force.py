"""Brute-force search for optimal refinement policy under some assumptions."""

import os

from marl_amr.envs.graph_env_h import GraphEnv
from marl_amr.tools import amr_utils

from multiprocessing import Process

import commentjson
import mfem.ser as mfem
import numpy as np


def search(n, include_derefine=False):
    """Search for optimal refinement policy given the assumptions below.

    Currently hard-coded for 2 time steps.
    Assumptions:
    1. Refine a contiguous region at each step
    2. Each step refines the same number of elements

    Outputs .mesh and .gf files

    Args:
        n: number of contiguous elements, including and to the right of, a chosen
            element to be refined
        include_derefine: Bool, if true then derefines all elements not selected
            for refinement

    Returns:
        list of element ids to be refined at each time step
    """
    with open('../envs/configs/advection_1d_href.json', 'r') as f:
        config_env = commentjson.load(f)['env']

    best_error = np.inf
    best_choices = []

    env = GraphEnv(config_env)
    obs, info = env.reset()

    # enumerate available choices
    t0_num_elements = len(obs[0])
    t0_num_choices = t0_num_elements - n + 1
    print('number of choices at t=0', t0_num_choices)
    # anisotropic h-ref in 1D bisects each element
    t1_num_elements = len(obs[0]) + n
    t1_num_choices = t1_num_elements - n + 1
    print('number of choices at t=1', t1_num_choices)

    for c1 in range(t0_num_choices):
        # print('c1', c1)
        for c2 in range(t1_num_choices):
            # print('c2', c2)
            choices = [c1, c2]
            obs, info = env.reset()
            t = 0
            while t < 2:
                idx_start_of_region = choices[t]
                obs, r, done, info = run_step(env, idx_start_of_region, n,
                                              include_derefine)
                t += 1

            if info['true_global_error'] < best_error:
                best_error = info['true_global_error']
                best_choices = choices

    # Run best choices and generate mesh
    print('Case n =', n, 'best choices of start of refinement region', best_choices)
    print('best final global error', best_error)

    run_best_choices(env, best_choices, n, include_derefine)


def run_best_choices(env, best_choices, n, include_derefine=False):
    """Runs one episode.

    Args:
        best_choices: list of indices indicating start of contiguous region
        n: number of contiguous elements to refine
    """
    obs, info = env.reset()
    r_total = 0
    t = 0
    amr_utils.output_mesh(env, os.getcwd(), t)
    while t < 2:
        idx_start_of_region = best_choices[t]
        # print('t', t)
        obs, r, done, info = run_step(env, idx_start_of_region, n,
                                      include_derefine)
        t += 1
        r_total += r
        amr_utils.output_mesh(env, os.getcwd(), t)

    print('Case n =', n, 'sum_of_dofs', info['sum_of_dofs'])
    print('r_total', r_total)
    print('Case n =', n, 'true global error', info['true_global_error'])
    print('estimated global error', info['global_error'])


def get_elem_id_sorted_by_x(env):
    """Returns an array of elem_id, sorted by increasing x."""
    num_elements = env.mesh.GetNE()
    centers = np.zeros((num_elements, 2))
    v = mfem.Vector()
    for elem_id in range(num_elements):
        env.mesh.GetElementCenter(elem_id, v)
        centers[elem_id, :] = v.GetDataArray()
    elem_id_sorted_by_x = np.argsort(centers[:, 0])

    return elem_id_sorted_by_x


def run_step(env, idx_start, n, include_derefine=False):
    """Constructs actions and runs env.step"""
    elem_id_sorted_by_x = get_elem_id_sorted_by_x(env)
    elem_id_to_refine = set(elem_id_sorted_by_x[
        idx_start: idx_start+n])

    actions = {}
    for agent_id, agent in env.agent_manager.agents.items():
        if agent.elem() in elem_id_to_refine:
            if include_derefine:
                actions[agent_id] = 2 # refine
            else:
                actions[agent_id] = 1 # refine
        else:
            actions[agent_id] = 0 # derefine

    obs, r, done, info = env.step(actions)
    r = r[list(r.keys())[0]] * env.mesh.GetNE()
    return obs, r, done, info


def search_multiprocess(list_n, deref):

    processes = []

    for n in list_n:
        p = Process(target=search, args=(n, deref))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=4,
                        help='number of contiguous elements including and to '
                        'the right of a chosen element to be h-refined.')
    parser.add_argument('--choices', type=str, default=None,
                        help='comma-separate list of indices that denote '
                        'the start of contiguous refinement region')
    parser.add_argument('--deref', action='store_true',
                        help='allow derefinement')
    parser.add_argument('--mp', action='store_true',
                        help='multiprocess')
    parser.add_argument('--list_n', type=str, default=None,
                        help='comma-separate list of numbers to be run in parallel')
    args = parser.parse_args()

    if args.choices is not None:
        with open('../envs/configs/advection_1d_href.json', 'r') as f:
            config_env = commentjson.load(f)['env']
        env = GraphEnv(config_env)
        choices = list(map(int, args.choices.split(',')))
        run_best_choices(env, choices, args.n)
    elif args.mp:
        list_n = list(map(int, args.list_n.split(',')))
        search_multiprocess(list_n, args.deref)
    else:
        search(args.n, args.deref)
