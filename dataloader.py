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
from torchaudio.transforms import FrequencyMasking, TimeMasking

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['name']] = row['index']
            line_count += 1
    return index_lookup

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, data_conf):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json

        self.melbins = data_conf.get('num_mel_bins')
        self.target_length = data_conf.get('sample_length')
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = data_conf.get('spec_mean')
        self.norm_std = data_conf.get('spec_std')
        self.noise = data_conf.get('add_noise')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = data_conf.get('skip_norm') 
        self.task_type = data_conf.get('task_type')
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('Initializing dataset with {} samples and {} classes'.format(len(self.data), self.label_num))
            
    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)
            
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        n_frames = fbank.shape[0]

        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            start_id = np.random.randint(0, -p)
            fbank = fbank[start_id:start_id+1024, :]
        
        if filename2 == None:
            return fbank
        else:
            return fbank, mix_lambda


    def __getitem__(self, index, sim_pair=True):
        """
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        """
        
        datum = self.data[index]
        if random.random() < self.mixup:
            mix_sample_idx = random.randing(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            label_indices = np.zeros(self.label_num)
            for label_str in datum['labels'].split(';'):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            for label_str in mix_datum['labels'].split(';'):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
        else:
            label_indices = np.zeros(self.label_num)
            fbank = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(';'):
                label_indices[int(self.index_dict[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)
        

        fbank = torch.transpose(fbank, 0, 1).unsqueeze(0)
        if self.freqm != 0:
            freqm = FrequencyMasking(self.freqm)
            fbank = freqm(fbank)
        if self.timem != 0:
            timem = TimeMasking(self.timem)
            fbank = timem(fbank)
        fbank = torch.transpose(fbank.squeeze(0), 0, 1)


        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        
        fbank = torch.transpose(fbank, 0, 1).unsqueeze(0)

        if self.task_type == 'classification':
            return fbank, label_indices
        else:
            return fbank, datum['labels']

    def __len__(self):
        return len(self.data)
