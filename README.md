## Sequence learning

fit_bidirdecay.py is the main script that loads the data and runs the fitting in parallel, and it depends on the ddHCRP_LM.py file that includes the model class. 

Set the hyperparameters n_repetitions and n_param_search_iter such that the code should finish in about 2-2.5 days.

## Running in Nyx cluster environment.

The cluster uses singularity containers to bring in application environments.
In this case, python and statistical packages are brought in using a popular container.
To download the singularity container, run: pull_python_sing.sh

## Python parallelism and batch job (as a start)

To run, log in to nyx.hpc.kyb.local and on the command line:

```
module purge
module load
module load singularity
cd sequence-learning/
./datascience-python_latest.sif python fit_bidirdecay.py
```
You'll be doing this on the login node, so don't run long jobs that use a lot of cores!

Much better is to run the program as a batch job. For this, we user slurm.

To submit a batch job, we create a script with special #tags that slurm uses.
A slurm batch script has beenn created in: nyx_single_node.job

To run the work as a batch job when logged onto l01.kyb.local:
```
cd sequence-learning/
sbatch nyx-single-node.slurm.sh
```
You can use sinfo to see the state of the cluster. squeue will show you the queued jobs.

Output will be files will be created in the directory where the job was submitted.

If you want to cancel the job, scancel <jobid>





