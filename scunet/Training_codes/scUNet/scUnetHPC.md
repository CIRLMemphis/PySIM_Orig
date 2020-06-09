Install PyTorch on HPC using miniconda:
<pre><code>module load miniconda/2.7.15/gcc.7.2.0
module load cuda10.1/blas/10.1.243
module load cuda10.1/toolkit/10.1.243
conda create -n torch-gpu-env torch-gpu
conda activate torch-gpu-env
</pre></code>

Install additional packages:
<pre><code>conda install -c anaconda scikit-image
conda install scikit-learn</pre></code>

Schedule the training script on HPC:
<pre><code>module load slurm
sbatch train.sh</pre></code>
