import os
from shutil import *

import torch
import numpy as np

class CKPT_Manager:
    def __init__(self, root_dir, model_name, max_files_to_keep = 10):
        self.root_dir = root_dir
        self.model_name = model_name
        self.max_files = max_files_to_keep

        self.ckpt_list = os.path.join(self.root_dir, 'checkpoints')

    def load_ckpt(self, network, by_score = True, name = None):
        # read file
        try:
            with open(self.ckpt_list, 'r') as file:
                lines = file.read().splitlines()
                file.close()
        except:
            return
        # get ckpt path
        if name is None:
            if by_score:
                file_name = lines[0].split(' ')[0]
            else:
                file_name = lines[-1].split(' ')[0]
        else:
            file_name = name

        file_path = os.path.join(self.root_dir, file_name)
        return network.load_state_dict(torch.load(file_path)), os.path.basename(file_name).split('.pytorch')[0]

    def save_ckpt(self, network, epoch, score):
        if type(epoch) == str:
            file_name = self.model_name + '_' + epoch + '.pytorch'
        else:
            file_name = self.model_name + '_' + '{:05d}'.format(epoch) + '.pytorch'
        save_path = os.path.join(self.root_dir, file_name)

        torch.save(network.state_dict(), save_path)

        # remove the most recently added line
        if os.path.exists(self.ckpt_list):
            with open(self.ckpt_list, 'r') as file:
                lines = file.read().splitlines()
                line_to_remove  = lines[-1]
                if line_to_remove not in lines[:-1]:
                    os.remove(os.path.join(self.root_dir, line_to_remove.split(' ')[0]))
                del(lines[-1])
                file.close()
            with open(self.ckpt_list, 'w') as file:
                for line in lines:
                    file.write(line + os.linesep)
                file.close()

        with open(self.ckpt_list, 'a') as file:
            line_to_add = file_name + ' ' + str(score) + os.linesep
            file.write(line_to_add) # for the new ckpt
            file.write(line_to_add) # for the most recent ckpt
            file.close()

        self._update_files()

    def _update_files(self):
        # read file
        with open(self.ckpt_list, 'r') as file:
            lines = file.read().splitlines()
            file.close()

        # sort by score
        line_recent = lines[-1]
        lines_prev = self._sort(lines[:-1])

        # delete ckpt
        while len(lines_prev) > self.max_files:
            line_to_remove = lines_prev[-1]
            if line_to_remove != line_recent:
                os.remove(os.path.join(self.root_dir, line_to_remove.split(' ')[0]))
            del(lines_prev[-1])

        # update ckpt list
        with open(self.ckpt_list, 'w') as file:
            for line in lines_prev:
                file.write(line + os.linesep)
            file.write(line_recent + os.linesep)
            file.close()

    def _sort(self, lines):
        scores = [float(score.split(' ')[1]) for score in lines]
        lines = [line for _, line in sorted(zip(scores, lines))]

        return lines

