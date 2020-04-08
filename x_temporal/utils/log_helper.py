import logging
import sys

import torch
from x_temporal.utils.dist_helper import is_master_proc


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return

    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)

    logger.addFilter(lambda record: is_master_proc())

    format_str = f'%(asctime)s-%(filename)s#%(lineno)d:%(message)s'
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def get_log_format(multi_class=False):
    if multi_class:
        return  ('Epoch: [{2}/{3}]\tIter: [{0}/{1}]\t'
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'mAP {mAP.val:.3f} ({mAP.avg:.3f})\t'
                 'LR {lr:.4f}')
    else:
        return  ('Epoch: [{2}/{3}]\tIter: [{0}/{1}]\t'
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                 'LR {lr:.4f}')
