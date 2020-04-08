LOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n1  --gres=gpu:0 --ntasks-per-node=1 --cpus-per-task=36 \
python -u  vid2img_sthv2.py
