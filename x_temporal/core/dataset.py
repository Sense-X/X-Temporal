import os
import random
import logging

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from numpy.random import randint
from decord import VideoReader
from decord import cpu

logger = logging.getLogger('global')



class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def mlabel(self):
        labels = torch.tensor([int(x)
                               for x in self._data[2].split(',')]).long()
        onehot = torch.FloatTensor(313)
        onehot.zero_()
        onehot[labels] = 1
        return onehot


class VideoDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, multi_class=False,
                 temporal_samples=1, reverse_samples=False, dense_sample_rate=2,
                 video_source=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.multi_class = multi_class
        self.dense_sample_rate = dense_sample_rate
        self.video_source = video_source

        # new args for test
        self.temporal_samples = temporal_samples
        self.reverse_samples = reverse_samples

        if self.dense_sample:
            logger.info('=> Using dense sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()


    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                filename = os.path.join(
                    self.root_path, directory, self.image_tmpl.format(idx))
                img = Image.open(filename).convert('RGB')
                return [img]
            except Exception as e:
                logger.info(e)
                logger.info(
                    'error loading image: %s' %
                    os.path.join(
                        self.root_path,
                        directory,
                        self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(
                    self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    filename = os.path.join(
                        self.root_path, directory, self.image_tmpl.format(idx))
                    flow = Image.open(filename).convert('RGB')
                except Exception:
                    logger.info('error loading flow file: %s' %
                                os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(
                        os.path.join(
                            self.root_path,
                            directory,
                            self.image_tmpl.format(1))).convert('RGB')
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:ample_ind/5d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        logger.info('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_range = self.num_segments * self.dense_sample_rate
            sample_pos = max(1, 1 + record.num_frames - sample_range)
            t_stride = self.dense_sample_rate
            start_idx = 0 if sample_pos == 1 else np.random.randint(
                0, sample_pos - 1)
            offsets = [
                (idx * t_stride + start_idx) %
                record.num_frames for idx in range(
                    self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (
                record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(
                    randint(
                        record.num_frames -
                        self.new_length +
                        1,
                        size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_range = self.num_segments * self.dense_sample_rate
            sample_pos = max(1, 1 + record.num_frames - sample_range)
            t_stride = self.dense_sample_rate
            if self.temporal_samples == 1:
                start_idx = 0 if sample_pos == 1 else sample_pos // 2
                offsets = [
                    (idx * t_stride + start_idx) %
                    record.num_frames for idx in range(
                    self.num_segments)]
            else:
                start_list = np.linspace(0, sample_pos - 1, num=self.temporal_samples, dtype=int)
                offsets = []
                for start_idx in start_list.tolist():
                    offsets += [(idx * t_stride + start_idx) %
                    record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            t_offsets = []
            tick = (record.num_frames - self.new_length + 1) / \
                float(self.num_segments)
            offsets = np.array([1 + int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
            t_offsets.append(offsets)

            average_duration = (
                record.num_frames - self.new_length + 1) // self.num_segments
            for i in range(self.temporal_samples - 1):
                offsets = np.multiply(list(range(self.num_segments)),
                                      average_duration) + randint(average_duration,
                                                                  size=self.num_segments)
                t_offsets.append(offsets + 1)

            t_offsets = np.stack(t_offsets).reshape(-1)
            return t_offsets

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_range = self.num_segments * self.dense_sample_rate
            sample_pos = max(1, 1 + record.num_frames - sample_range)
            t_stride = self.dense_sample_rate
            start_list = np.linspace(0, sample_pos - 1, num=self.temporal_samples, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) %
                            record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            t_offsets = []
            tick = (record.num_frames - self.new_length + 1) / \
                float(self.num_segments)
            offsets = np.array([1 + int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
            t_offsets.append(offsets)

            average_duration = (
                record.num_frames - self.new_length + 1) // self.num_segments
            for i in range(self.temporal_samples - 1):
                offsets = np.multiply(list(range(self.num_segments)),
                                      average_duration) + randint(average_duration,
                                                                  size=self.num_segments)
                t_offsets.append(offsets + 1)

            t_offsets = np.stack(t_offsets).reshape(-1)
            return t_offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        # check this is a legit video folder
        if self.video_source:
            full_path = os.path.join(self.root_path, record.path)
            while not os.path.exists(full_path):
                logger.info(
                    '################## Not Found: %s' %
                    os.path.join(
                        self.root_path,
                        record.path))
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
                full_path = os.path.join(self.root_path, record.path)
        else:
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(
                    self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(
                    self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(
                    self.root_path, record.path, file_name)
            while not os.path.exists(full_path):
                logger.info(
                    '################## Not Found: %s' %
                    os.path.join(
                        self.root_path,
                        record.path,
                        file_name))
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
                if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                    file_name = self.image_tmpl.format('x', 1)
                    full_path = os.path.join(
                        self.root_path, record.path, file_name)
                elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                    file_name = self.image_tmpl.format(
                        int(record.path), 'x', 1)
                    full_path = os.path.join(
                        self.root_path, '{:06d}'.format(int(record.path)), file_name)
                else:
                    file_name = self.image_tmpl.format(1)
                    full_path = os.path.join(
                        self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices, record.path)

    def get(self, record, indices, path):
        images = list()
        if not self.video_source:
            for seg_ind in indices:
                p = int(seg_ind)
                seg_imgs = self._load_image(path, p)
                images.extend(seg_imgs)
        else:
            vr = VideoReader(
                os.path.join(
                    self.root_path,
                    record.path),
                ctx=cpu(0))
            for seg_ind in indices:
                try:
                    images.append(Image.fromarray(vr[seg_ind-1].asnumpy()))
                except Exception as e:
                    images.append(Image.fromarray(vr[0].asnumpy()))

        process_data = self.transform(images)
        if self.multi_class:
            return process_data, record.mlabel
        else:
            return process_data, record.label

    def __len__(self):
        return len(self.video_list)
