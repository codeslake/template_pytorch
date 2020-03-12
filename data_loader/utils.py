import os
import numpy as np
import cv2
from pathlib import Path

def read_frame(path):
    # frame = scipy.misc.imread(path, mode='RGB')
    # frame = frame - 127.5
    # frame = frame / (255. / 2.)

    frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    frame = frame / 255.

    return np.expand_dims(frame, axis = 0)

def crop_multi(x, wrg, hrg, is_random=False, row_index=0, col_index=1):
  
    h, w = x[0].shape[row_index], x[0].shape[col_index]

    if (h <= hrg) or (w <= wrg):
        raise AssertionError("The size of cropping should smaller than the original image")

    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
        results = []
        for data in x:
            results.append(data[int(h_offset):int(hrg + h_offset), int(w_offset):int(wrg + w_offset)])
        return np.asarray(results)
    else:
        # central crop
        h_offset = (h - hrg) / 2
        w_offset = (w - wrg) / 2
        results = []
        for data in x:
            results.append(data[int(h_offset):int(h - h_offset), int(w_offset):int(w - w_offset)])
        return np.asarray(results)

def get_dict_array_by_key(key, array_num):
    data_holder = collections.OrderedDict()
    for holder_name in key:
        data_holder[holder_name] = [None] * array_num

    return data_holder

def load_file_list(root_path, child_path = None):
    folder_paths = []
    filenames_pure = []
    filenames_structured = []
    num_files = 0
    for root, dirnames, filenames in os.walk(root_path):
        if len(dirnames) != 0:
            if dirnames[0][0] == '@':
                del(dirnames[0])

        if len(dirnames) == 0:
            if root[0] == '.':
                continue
            if child_path is not None and child_path != Path(root).name: 
                continue
            folder_paths.append(root)
            filenames_pure = []
            for i in np.arange(len(filenames)):
                if filenames[i][0] != '.' and filenames[i] != 'Thumbs.db':
                    filenames_pure.append(os.path.join(root, filenames[i]))
            filenames_structured.append(np.array(sorted(filenames_pure)))
            num_files += len(filenames_pure)

    folder_paths = np.array(folder_paths)
    filenames_structured = np.array(filenames_structured)

    sort_idx = np.argsort(folder_paths)
    folder_paths = folder_paths[sort_idx]
    filenames_structured = filenames_structured[sort_idx]

    return folder_paths, np.squeeze(filenames_structured), np.squeeze(num_files)

def get_base_name(path):
    return os.path.basename(path.split('.')[0])

def get_folder_name(path):
    path = os.path.dirname(path)
    return path.split(os.sep)[-1]

