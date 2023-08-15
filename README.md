# Multi-Agent Reinforcement Learning for Adaptive Mesh Refinement

MARL-AMR is the code for the paper [Multi-Agent Reinforcement Learning for Adaptive Mesh Refinement](https://arxiv.org/abs/2211.00801), published at AAMAS 2023. It contains the implementation of a new algorithm, called Value Decomposition Graph Network (VDGN), for applying multi-agent reinforcement learning to the problem of adaptive mesh refinement (AMR). It also contains the implementation of a multi-agent environment for AMR on a linear advection problem. VDGN is the first learning algorithm to display anticipatory refinement behavior in AMR, and it outperforms local error threshold-based heuristic strategies.


## Setup

Basic requirements:
- Python 3.6
- GCC 6.1
- Swig 4.0.2


### Install SWIG

SWIG is required for installing the finite element code. To install it, follow instructions [here](https://www.linuxfromscratch.org/blfs/view/svn/general/swig.html).
Use `--prefix=<optional local directory>` if you do not have root access.


### Compile MFEM, and install PyMFEM and this repo in a virtualenv

From this directory, run `$ ./install.sh`. This will create a build directory called `amr_build` one level above this directory to house MFEM, PyMFEM, and a python virtualenv called `amr_env`.


## Training and evaluation

- Train VDGN using the config in `marl_amr/alg/configs/advection_vdgn.py`. Models and training logs will be saved in `marl_amr/results/advection/`.
```
$ source ../amr_build/amr_env/bin/activate
$ cd marl_amr/alg
$ python train_offpolicy.py --config_name=advection_vdgn
```

- Test a trained policy that is provided in this repo. The model checkpoint is located at `marl_amr/results/nx16_ny16_depth1_tstep0p25_vdgn_pretrained`.
```
$ cd marl_amr/alg
$ python test.py tf advection_test
```

- To save the meshes for visualization by [GLVis](https://glvis.org/), run with the option `--save_mesh_all_steps`. Mesh files will be saved in `marl_amr/results/mesh_files/`.
```
$ python test.py tf --save_mesh_all_steps
```


## Contributing

Please submit any bugfixes or feature improvements as [pull requests](https://help.github.com/articles/using-pull-requests/)


## Reference

If you find this code useful for your work, please cite this paper:
<pre>
@inproceedings{yang2023multi,
  title={Multi-Agent Reinforcement Learning for Adaptive Mesh Refinement},
  author={Yang, Jiachen and Mittal, Ketan and Dzanic, Tarik and Petrides, Socratis and Keith, Brendan and Petersen, Brenden and Faissol, Daniel and Anderson, Robert},
  booktitle={Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
  pages={14--22},
  year={2023}
}
</pre>


## License

MARL-AMR is distributed under the terms of the BSD-3 license. All new contributions must be made under this license.

See [LICENSE](LICENSE) and [NOTICE](NOTICE).

SPDX-License-Identifier: BSD-3

LLNL-CODE-853184
