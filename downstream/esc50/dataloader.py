# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py.py

# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import pickle

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, dataset_pkl_file, label_csv, max_len=20):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        self.data = pickle.load(open(dataset_pkl_file, 'rb'))
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.metadata = data_json['data']

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        self.max_len = max_len
#        ids = [x['ytid'] for x in self.metadata]
        self.metadata = [x for x in self.metadata if x['wav'].split('/')[-1].split('.wav')[0] in self.data.keys()]
#        self.data = {k:v for k,v in self.data.items() if k in ids}
        
        print('number of samples is {}, number of classes is {}'.format(len(self.metadata), self.label_num))

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.metadata[index]
        feats = self.data[datum['wav'].split('/')[-1].split('.wav')[0]]
        feats = torch.FloatTensor(feats)
        
        label_indices = np.zeros(self.label_num)
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0
        label_indices = torch.FloatTensor(label_indices)

        return feats, label_indices


    def __len__(self):
        return len(self.metadata)
     
