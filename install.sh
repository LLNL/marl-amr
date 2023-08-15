#!/bin/sh

# Define root as one level up from this project dir
cd ..
repo_root=$(pwd)

# Create a build directory that will contain MFEM, PyMFEM, and virtualenv
mkdir amr_build
cd $repo_root/amr_build

# Compile the drl4amr-advection branch of MFEM
echo "Cloning MFEM"
git clone -b drl4amr-advection https://github.com/mfem/mfem.git
cd mfem
git checkout 4127f77bbf4d04680b3f9193ac30e09b7c23d2a9
echo "Compiling MFEM"
make serial MFEM_SHARED=YES -j
make install -j

# Create virtualenv called "amr_env"
echo "Creating virtualenv"
cd $repo_root/amr_build
python3.6 -m venv amr_env
source amr_env/bin/activate
pip install --upgrade pip

# Install dependencies
echo "Installing python dependencies"
cd $repo_root/marl-amr
pip install -r requirements.txt

# Clone PyMFEM and install
echo "Cloning PyMFEM"
cd $repo_root/amr_build
git clone -b drl4amr https://github.com/mfem/PyMFEM.git
cd PyMFEM
git checkout 44bda2efadab39a15225532e4cff62bbcfc38ac0
echo "Installing PyMFEM"
mfem_prefix="${repo_root}/amr_build/mfem/mfem/"
mfem_source="${repo_root}/amr_build/mfem/"
python setup.py install --mfem-prefix=$mfem_prefix --mfem-source=$mfem_source

# Install project
echo "Installing marl-amr"
cd $repo_root/marl-amr
pip install -e .

# Check installation
echo "Checking installation (expect to see 'advection')"
result=$(python -c "from marl_amr.envs.solvers.AdvectionSolver import AdvectionSolver; print(AdvectionSolver.name)")
if [ "$result" == "advection" ]; then
    echo "Installation finished successfully"
fi
