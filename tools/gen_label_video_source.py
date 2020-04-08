import os

from decord import VideoReader
from decord import cpu

if __name__ == '__main__':
    root_dir = ''  # video data root path
    dataset_name = 'hmdb51'
    with open(os.path.join('../datasets', dataset_name, 'category.txt')) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)


    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    filename_input = os.path.join('../datasets', dataset_name, 'vallist.txt')
    filename_output = 'test_videofolder.txt'
    with open(filename_input) as f:
        lines = f.readlines()
    videos = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        videos.append(line)
        label = line.split('/')[0]
        idx_categories.append(dict_categories[label])
    output = []
    for i in range(len(videos)):
        curVideo = videos[i]
        curIDX = idx_categories[i]
        video_file = os.path.join(root_dir, curVideo)
        vr = VideoReader(os.path.join(root_dir, curVideo), ctx=cpu(0))
        output.append('%s %d %d' % (curVideo, len(vr), curIDX))
        print('%d/%d' % (i, len(vr)))
    with open(filename_output, 'w') as f:
        f.write('\n'.join(output))
