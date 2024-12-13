#!/bin/bash -l


#SBATCH --output=my_job_output.out  # Output file
#SBATCH --error=my_job_error.err    # Error file
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00




# source /opt/python/conda/2020.07_py3.8/anaconda/etc/profile.d/conda.sh
module load python/3.11
module load mpi4py/3.1.4
module load scipy-stack/2024a
# source .../ENV/bin/activate

# conda activate my_mpi_env
mpirun -n 20  python3 ../kheya_test.py