import os


dataset_path = 'kinetics_600/train_frames'
label_path = ''

if __name__ == '__main__':
    with open('kinetics_label_map.txt') as f:
        categories = f.readlines()
        categories = [
            c.strip().replace(
                ' ',
                '_').replace(
                '"',
                '').replace(
                '(',
                '').replace(
                    ')',
                    '').replace(
                        "'",
                '') for c in categories]
    assert len(set(categories)) == 600
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    print(dict_categories)

    files_input = ['kinetics-600_val.csv', 'kinetics-600_train.csv']
    files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        count_cat = {k: 0 for k in dict_categories.keys()}
        with open(os.path.join(label_path, filename_input)) as f:
            lines = f.readlines()[1:]
        folders = []
        idx_categories = []
        categories_list = []
        for line in lines:
            line = line.rstrip()
            items = line.split(',')
            st = int(items[2])
            et = int(items[3])
            folders.append(items[1] + '_' + "%06d" % st + '_' + "%06d" % et)
            this_catergory = items[0].replace(
                ' ',
                '_').replace(
                '"',
                '').replace(
                '(',
                '').replace(
                ')',
                '').replace(
                    "'",
                '')
            categories_list.append(this_catergory)
            idx_categories.append(dict_categories[this_catergory])
            count_cat[this_catergory] += 1
        print(max(count_cat.values()))

        assert len(idx_categories) == len(folders)
        missing_folders = []
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            img_dir = os.path.join(dataset_path, categories_list[i], curFolder)
            if not os.path.exists(img_dir):
                missing_folders.append(img_dir)
                # print(missing_folders)
            else:
                dir_files = os.listdir(img_dir)
                output.append(
                    '%s %d %d' %
                    (os.path.join(
                        categories_list[i],
                        curFolder),
                        len(dir_files),
                        curIDX))
            print(
                '%d/%d, missing %d' %
                (i, len(folders), len(missing_folders)))
        with open(os.path.join(label_path, filename_output), 'w') as f:
            f.write('\n'.join(output))
        with open(os.path.join(label_path, 'missing_' + filename_output), 'w') as f:
            f.write('\n'.join(missing_folders))
