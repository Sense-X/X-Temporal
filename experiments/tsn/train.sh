T=`date +%m%d%H%M`
ROOT=../..
cfg=default.yaml

export PYTHONPATH=$ROOT:$PYTHONPATH

python $ROOT/x_temporal/train.py --config $cfg | tee log.train.$T
