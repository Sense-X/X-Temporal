# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V1

import os

if __name__ == '__main__':
    dataset_name = 'something-something-v1'  # 'jester-v1'
    with open('%s-labels.csv' % dataset_name) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open('category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    '''
    files_input = ['%s-validation.csv' % dataset_name, '%s-train.csv' % dataset_name]
    files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join('../something/v1/20bn-something-something-v1', curFolder))
            output.append('%s %d %d' % ('' + curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
    '''
    filename_input = '%s-test.csv' % dataset_name
    filename_output = 'test_videofolder.txt'
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        folders.append(line)
        idx_categories.append(0)
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        dir_files = os.listdir(
            os.path.join(
                '../something/v1/20bn-something-something-v1',
                curFolder))
        output.append('%s %d %d' % ('' + curFolder, len(dir_files), curIDX))
        print('%d/%d' % (i, len(folders)))
    with open(filename_output, 'w') as f:
        f.write('\n'.join(output))
