import argparse
import yaml
from easydict import EasyDict
import torch

from x_temporal.interface.temporal_helper import TemporalHelper
from x_temporal.utils.multiprocessing import mrun


parser = argparse.ArgumentParser(description='X-Temporal')
parser.add_argument('--config', type=str, help='the path of config file')
parser.add_argument("--shard_id", help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0, type=int)
parser.add_argument("--num_shards", help="Number of shards using by the job",
        default=1, type=int)
parser.add_argument("--init_method", help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999", type=str)
parser.add_argument('--dist_backend', default='nccl', type=str)

def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config['config'])
    if config.gpus > 1:
        torch.multiprocessing.spawn(
                mrun,
                nprocs=config.gpus,
                args=(config.gpus,
                    args.init_method,
                    args.shard_id,
                    args.num_shards,
                    args.dist_backend,
                    config,
                    'test',
                    ),
                daemon=False)
    else:
        temporal_helper = TemporalHelper(config, inference_only=True)
        temporal_helper.evaluate()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("forkserver")
    main()
