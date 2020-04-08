import logging
import copy
import math
import os
import json
import shutil
import time


import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from x_temporal.utils.log_helper import init_log, get_log_format
from x_temporal.utils.lr_helper import build_scheduler
from x_temporal.utils.optimizer_helper import build_optimizer
from x_temporal.utils.metrics import Top1Metric
from x_temporal.utils.utils import format_cfg, accuracy, AverageMeter, load_checkpoint
from x_temporal.utils.dist_helper import (get_rank, get_world_size, all_gather, all_reduce)
from x_temporal.utils.model_helper import load_state_dict
from x_temporal.utils.dataset_helper import get_val_crop_transform, get_dataset, shuffle_dataset
from x_temporal.core.models_entry import get_model, get_augmentation
from x_temporal.core.transforms import *
from x_temporal.core.dataset import VideoDataSet


class TemporalHelper(object):
    def __init__(self, config, work_dir='./', ckpt_dict=None, inference_only=False):
        """
        Args:
            config: configuration for training and testing, sometimes
        """
        self.work_dir = work_dir
        self.inference_only = inference_only
        self.config = copy.deepcopy(config)

        self._setup_env()
        self._init_metrics()
        self._build()
        self._resume(ckpt_dict)
        self._ready()
        self._last_time = time.time()
        self.logger.info('Running with config:\n{}'.format(format_cfg(self.config)))
    
    def _resume(self, ckpt=None):
        """load state from given checkpoint or from pretrain_model/resume_model
        """
        if ckpt is None:
            ckpt = self.load_pretrain_or_resume()
            if ckpt is None:
                self.logger.info('Train from scratch...')
                return

        # load model weights
        load_state_dict(self.model, ckpt['model'], strict=False)
        if not self.inference_only:
            if 'optimizer' in ckpt:
                self.start_iter = ckpt['epoch'] * self.epoch_iters
                self.cur_epoch = self.start_epoch = ckpt['epoch']
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                self.logger.info(f'resume from epoch:{self.start_epoch}')

    def _build(self):
        self.data_loaders = self._build_dataloaders()
        self.model = self.build_model()
        self.criterion = self.build_criterion()
        
        if not self.inference_only:
            self.optimizer = build_optimizer(self.config.trainer.optimizer, self.model)
            self.lr_scheduler = build_scheduler(
                    self.config.trainer['lr_scheduler'],
                    self.optimizer, self.epoch_iters, self.world_size * self.config.dataset.batch_size)

    def build_criterion(self):
        if self.config.trainer.loss_type == 'nll':
            criterion = torch.nn.CrossEntropyLoss()
        elif self.config.trainer.loss_type == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown loss type")
        return criterion


    def _build_dataloaders(self):
        dataloader_dict = {}
        data_types = ['val', 'test']
        if not self.inference_only:
            data_types.append('train')
        for data_type in data_types:
            if data_type in self.config.dataset:
                dataloader_dict[data_type] = self._build_dataloader(data_type)
        return dataloader_dict

    def _build_dataloader(self, data_type):
        dargs = self.config.dataset
        if dargs.modality == 'RGB':
            data_length = 1
        elif dargs.modality in ['Flow', 'RGBDiff']:
            data_length = 5

        if dargs.modality != 'RGBDiff':
            normalize = GroupNormalize(dargs.input_mean, dargs.input_std)
        else:
            normalize = IdentityTransform()


        if data_type == 'train':
            train_augmentation = get_augmentation(self.config)
            transform = torchvision.transforms.Compose([
                train_augmentation,
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
                ConvertDataFormat(self.config.net.model_type),
            ])
            dataset = get_dataset(dargs, data_type, False, transform, data_length)
            self.train_data_size = len(dataset)
            self.epoch_iters =  math.ceil(float(self.train_data_size) /dargs.batch_size / self.world_size)
            self.max_iter = self.epoch_iters * self.config.trainer.epochs
            sampler = DistributedSampler(dataset) if self.config.gpus  > 1 else None

            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=dargs.batch_size, shuffle=(False if sampler else True),
                num_workers=dargs.workers, pin_memory=True,
                drop_last=True, sampler=sampler)
            return train_loader

        else:
            if self.inference_only:
                spatial_crops = self.config.get('evaluate', {}).get('spatial_crops', 1)
                temporal_samples = self.config.get('evaluate', {}).get('temporal_samples', 1)
            else:
                spatial_crops = 1
                temporal_samples = 1

            crop_aug = get_val_crop_transform(self.config.dataset, spatial_crops)
            transform = torchvision.transforms.Compose([
                GroupScale(int(dargs.scale_size)),
                crop_aug,
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
                ConvertDataFormat(self.config.net.model_type),
            ])

            dataset = get_dataset(dargs, data_type, True, transform, data_length, temporal_samples)
            sampler = DistributedSampler(dataset) if self.config.gpus  > 1 else None
            val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=dargs.batch_size, shuffle=(False if sampler else False), 
                drop_last=False, num_workers=dargs.workers, 
                pin_memory=True, sampler=sampler)
            return val_loader


    def _setup_env(self):

        # set random seed
        np.random.seed(self.config.get('seed', 2020))
        torch.manual_seed(self.config.get('seed', 2020))

        init_log('global', logging.INFO)
        self.logger = logging.getLogger('global')
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.start_iter = self.cur_iter = 0
        self.cur_epoch = 0
        self.best_prec1 = 0.0
        self.multi_class = self.config.dataset.get('multi_class', False)
        if self.multi_class:
            from x_temporal.utils.calculate_map import calculate_mAP
            self.calculate_mAP = calculate_mAP
        if self.rank == 0 and not self.inference_only:
            self.tb_logger = SummaryWriter(os.path.join(self.work_dir, 'events'))
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)

            if not os.path.exists(os.path.join(self.work_dir, self.config.saver.save_dir)):
                os.makedirs(os.path.join(self.work_dir, self.config.saver.save_dir))

    def _ready(self):
        self.model = self.model.cuda()
    
    def build_model(self):
        model = get_model(self.config).cuda()
        return model

    def get_dump_dict(self):
        return {
                'epoch': self.cur_epoch,
                'optimizer': self.optimizer.state_dict(),
                'model': self.model.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'best_prec1': self.best_prec1
                }

    def get_batch(self, batch_type='train'):
        assert batch_type in self.data_loaders
        if not hasattr(self, 'data_iterators'):
            self.data_iterators = {}
        if batch_type not in self.data_iterators:
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
        else:
            iterator = self.data_iterators[batch_type]

        try:
            batch = next(iterator)
        except StopIteration as e:  # noqa
            shuffle_dataset(self.data_loaders[batch_type], self.cur_epoch)
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
            batch = next(iterator)
	
        batch[0] = batch[0].cuda(non_blocking=True)
        batch[1] = batch[1].cuda(non_blocking=True)

        return batch

    def get_total_iter(self):
        return self.max_iter

    @staticmethod
    def load_weights(model, ckpt):
        assert 'model' in ckpt or 'state_dict' in ckpt
        model.load_state_dict(ckpt.get('model', ckpt.get('state_dict', {})), False)


    def forward(self, batch):
        data_time = time.time() - self._last_time
        output = self.model(batch[0])
        loss = self.criterion(output, batch[1])
        if self.multi_class:
            mAP = self.calculate_mAP(output, batch[1])
            self._preverse_for_show = [loss.detach(), data_time, mAP]
        else:
            prec1, prec5 = accuracy(output, batch[1], topk=(1, 5))
            self._preverse_for_show = [loss.detach(), data_time, prec1.detach(), prec5.detach()]
        return loss

    def backward(self, loss):
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        return loss

    def update(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        batch_time = time.time() - self._last_time
        if self.multi_class:
            loss, data_time, mAP = self._preverse_for_show
            self.reduce_update_metrics(loss, data_time, batch_time, mAP=mAP)
        else:
            loss, data_time, top1, top5 = self._preverse_for_show
            self.reduce_update_metrics(loss, data_time, batch_time, prec1=top1, prec5=top5)
        self._last_time = time.time()

    def _init_metrics(self):
        self.metrics = {}
        self.metrics['losses'] = AverageMeter(self.config.trainer.print_freq)
        self.metrics['batch_time'] = AverageMeter(self.config.trainer.print_freq)
        self.metrics['data_time'] = AverageMeter(self.config.trainer.print_freq)
        if self.multi_class:
            self.metrics['mAP'] = AverageMeter(self.config.trainer.print_freq)
        else:
            self.metrics['top1'] = AverageMeter(self.config.trainer.print_freq)
            self.metrics['top5'] = AverageMeter(self.config.trainer.print_freq)

    def reduce_update_metrics(self, loss, data_time, batch_time, prec1=None, prec5=None, mAP=None):
        reduced_loss = loss.clone()
        if self.config.gpus > 1:
            all_reduce(reduced_loss)

        self.metrics['losses'].update(reduced_loss.item())
        self.metrics['batch_time'].update(batch_time)
        self.metrics['data_time'].update(data_time)

        if self.multi_class:
            reduced_mAP = torch.Tensor([mAP]).cuda()
            if self.config.gpus > 1:
                all_reduce(reduced_mAP)
            self.metrics['mAP'].update(reduced_mAP.item())
        else:
            reduced_prec1 = prec1.clone()
            reduced_prec5 = prec5.clone()
            if self.config.gpus > 1:
                all_reduce(reduced_prec1)
                all_reduce(reduced_prec5)
            self.metrics['top1'].update(reduced_prec1.item())
            self.metrics['top5'].update(reduced_prec5.item())
    
    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key].reset()

    def train(self):
        self.model.cuda().train()
        for iter_idx in range(self.start_iter, self.max_iter):
            self.cur_epoch = int(float(iter_idx + 1) / self.epoch_iters)
            self.cur_iter = iter_idx
            inputs = self.get_batch('train')
            loss = self.forward(inputs)

            self.backward(loss)
            if self.config.trainer.clip_gradient > 0:
                clip_grad_norm_(self.model.parameters(), self.config.trainer.clip_gradient)
            self.update()

            if iter_idx % self.config.trainer.print_freq == 0 and self.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.metrics['losses'].avg, iter_idx)
                self.tb_logger.add_scalar('lr', self.lr_scheduler.get_lr()[0], iter_idx)
                log_formatter = get_log_format(self.multi_class)
                if self.multi_class:
                    self.tb_logger.add_scalar('mAP_train', self.metrics['mAP'].avg, iter_idx)
                    self.logger.info(log_formatter.format(
                        iter_idx, self.max_iter, self.cur_epoch + 1, self.config.trainer.epochs, 
                        batch_time=self.metrics['batch_time'], data_time=self.metrics['data_time'], loss=self.metrics['losses'], 
                        mAP=self.metrics['mAP'], lr=self.lr_scheduler.get_lr()[0]))
                else:
                    self.tb_logger.add_scalar('acc1_train', self.metrics['top1'].avg, iter_idx)
                    self.tb_logger.add_scalar('acc5_train', self.metrics['top5'].avg, iter_idx)
                    self.logger.info(log_formatter.format(
                        iter_idx, self.max_iter, self.cur_epoch + 1, self.config.trainer.epochs, 
                        batch_time=self.metrics['batch_time'], data_time=self.metrics['data_time'], loss=self.metrics['losses'], 
                        top1=self.metrics['top1'], top5=self.metrics['top5'], lr=self.lr_scheduler.get_lr()[0]))

            if (iter_idx == self.max_iter - 1) or (iter_idx % self.epoch_iters == 0 and iter_idx > 0 and \
                    self.cur_epoch % self.config.trainer.eval_freq == 0):
                metric  = self.evaluate()

                if self.rank == 0 and self.tb_logger is not None:
                    self.tb_logger.add_scalar('loss_val', metric.loss, iter_idx)
                    if self.multi_class:
                        self.tb_logger.add_scalar('mAP_val', metric.top1, iter_idx)
                    else:
                        self.tb_logger.add_scalar('acc1_val', metric.top1, iter_idx)
                        self.tb_logger.add_scalar('acc5_val', metric.top5, iter_idx)

                if self.rank == 0:
                    # remember best prec@1 and save checkpoint
                    is_best = metric.top1 > self.best_prec1
                    self.best_prec1 = max(metric.top1, self.best_prec1)
                    self.save_checkpoint({
                        'epoch': self.cur_epoch,
                        'optimizer': self.optimizer.state_dict(),
                        'model': self.model.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict(),
                        'best_prec1': self.best_prec1
                    }, is_best)

                if self.multi_class:
                    self.logger.info(' * Best mAP {:.3f}'.format(self.best_prec1))
                else:        
                    self.logger.info(' * Best Prec@1 {:.3f}'.format(self.best_prec1))


            end = time.time()

    def save_checkpoint(self, state, is_best):
        torch.save(state, os.path.join(self.work_dir, self.config.saver.save_dir, 'ckpt.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(self.work_dir, self.config.saver.save_dir, 'ckpt.pth.tar'), 
                    os.path.join(self.work_dir, self.config.saver.save_dir, 'ckpt_best.pth.tar'))

    @torch.no_grad()
    def evaluate(self):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        if self.multi_class:
            mAPs = AverageMeter(0)
        else:
            top1 = AverageMeter(0)
            top5 = AverageMeter(0)

        if self.inference_only:
            spatial_crops = self.config.get('evaluate', {}).get('spatial_crops', 1)
            temporal_samples = self.config.get('evaluate', {}).get('temporal_samples', 1)
        else:
            spatial_crops = 1
            temporal_samples = 1
        dup_samples = spatial_crops * temporal_samples

        self.model.cuda().eval()
        test_loader = self.data_loaders['test']
        test_len = len(test_loader)
        end = time.time()
        for iter_idx in range(test_len):
            inputs = self.get_batch('test')
            isizes = inputs[0].shape

            if self.config.net.model_type == '2D':
                inputs[0] = inputs[0].view(
                    isizes[0] * dup_samples, -1, isizes[2], isizes[3])
            else:
                inputs[0] = inputs[0].view(
                    isizes[0], isizes[1], dup_samples, -1, isizes[3], isizes[4]
                        )
                inputs[0] = inputs[0].permute(0, 2, 1, 3, 4, 5).contiguous()
                inputs[0] = inputs[0].view(isizes[0] * dup_samples, isizes[1], -1, isizes[3], isizes[4])

            output = self.model(inputs[0])
            osizes = output.shape

            output = output.view((osizes[0] // dup_samples, -1, osizes[1]))
            output = torch.mean(output, 1)


            loss = self.criterion(output, inputs[1])
            num = inputs[0].size(0)
            losses.update(loss.item(), num)
            if self.multi_class:
                mAP = self.calculate_mAP(output, inputs[1])
                mAPs.update(mAP, num)
            else:
                prec1, prec5 = accuracy(output, inputs[1], topk=(1, 5))
                top1.update(prec1.item(), num)
                top5.update(prec5.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if iter_idx % self.config.trainer.print_freq == 0:
                self.logger.info('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                       iter_idx, test_len, batch_time=batch_time))

        total_num = torch.Tensor([losses.count]).cuda()
        loss_sum = torch.Tensor([losses.avg*losses.count]).cuda()

        if self.config.gpus > 1:
            all_reduce(total_num, False)
            all_reduce(loss_sum,  False)
        final_loss = loss_sum.item()/total_num.item()
        
        if self.multi_class:
            mAP_sum = torch.Tensor([mAPs.avg*mAPs.count]).cuda()
            if self.config.gpus > 1:
                all_reduce(mAP_sum)
            final_mAP = mAP_sum.item()/total_num.item()
            self.logger.info(' * mAP {:.3f}\tLoss {:.3f}\ttotal_num={}'.format(final_mAP, final_loss,
                total_num.item()))
            metric = Top1Metric(final_mAP, 0, final_loss)
        else:
            top1_sum = torch.Tensor([top1.avg*top1.count]).cuda()
            top5_sum = torch.Tensor([top5.avg*top5.count]).cuda()
            if self.config.gpus > 1:
                all_reduce(top1_sum, False)
                all_reduce(top5_sum, False)
            final_top1 = top1_sum.item()/total_num.item()
            final_top5 = top5_sum.item()/total_num.item()
            self.logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\ttotal_num={}'.format(final_top1,
                final_top5, final_loss, total_num.item()))
            metric = Top1Metric(final_top1, final_top5, final_loss)

        self.model.cuda().train()
        return metric

    def load_pretrain_or_resume(self):
        if 'resume_model' in self.config.saver:
            self.logger.info('Load checkpoint from {}'.format(self.config.saver['resume_model']))
            return load_checkpoint(self.config.saver['resume_model'])
        elif 'pretrain_model' in self.config.saver:
            state = load_checkpoint(self.config.saver['pretrain_model'])
            self.logger.info('Load checkpoint from {}'.format(self.config.saver['pretrain_model']))
            return {'model': state['model']}
        else:
            self.logger.info('Load nothing! No weights provided {}')
            return None

