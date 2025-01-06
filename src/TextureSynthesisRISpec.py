import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as func
import torchaudio.functional as F
import torchaudio.transforms as T

class TextureSynthesisRISpec(nn.Module):
    def __init__(self,
                 ref_waveform,
                 out_chan,
                 device,
                 filter_sizes = [(101, 2), (53, 3), (11, 5), (3, 3),
                                 (5, 5), (11, 11), (19, 19),(27, 27)]):
        '''
        ref_waveform: reference waveform
        out_chan: number of output channels from each shallow CNN
        device: cuda or not
        filter_sizes: list of filter sizes

        '''
        super(TextureSynthesisRISpec, self).__init__()
        #convolutions for shallow CNN
        self.convs = []
        for filter_size in filter_sizes:
             self.convs.append(nn.Conv2d(in_channels=2,
			      out_channels=out_chan,
			      kernel_size=filter_size,
                              stride=(1,1),
                              padding='valid',
                              bias = False))
             self.convs[-1].weight.data = self.convs[-1].weight.data.uniform_(-0.05, 0.05).to(device)
        self.relu = nn.ReLU()
        self.device = device

        #reference RI spec if needed
        self.ref_stats = []
        RI_spec = self.get_RI_spec(waveform)
        CNN_outputs = self.apply_random_CNN(RI_spec)
        for out in CNN_outputs:
            self.ref_stats.append(self.calc_cov(out).detach())

    def get_RI_spec(self, waveform, n_fft_for_stft = 512,
                                    hop_length_for_stft = 256,
                                    window_length = 512):
        '''
        gets RI spectogram of waveform
        '''
        waveform_stft = torch.stft(waveform,
                      n_fft = n_fft_for_stft,
                      hop_length = hop_length_for_stft,
                      window = None,
                      return_complex=True)
        waveform_stft_normalized = waveform_stft/torch.max(torch.abs(waveform_stft))
        waveform_stft_real = torch.view_as_real(waveform_stft_normalized)
        RI_spec_unpermuted = 2 * func.sigmoid(10 * waveform_stft_real) - 1
        RI_spec = torch.permute(RI_spec_unpermuted, (0,3,1,2))
        return RI_spec

    def apply_random_CNN(self, RI_spec):
        '''
        applies random CNNs to RI spec for feature extraction
        '''
        output = []
        for conv in self.convs:
             #squeeze ensures everything is 3d
             output.append(self.relu(conv(RI_spec)).squeeze())
        return output

    def calc_cov(self, tensor):
        """flattens tensor and calculates sample mean and covariance matrix
        along channel"""
        tensor = tensor.squeeze()
        return torch.matmul(tensor, torch.transpose(tensor, 2,1))

    def get_gram_matrix(self, waveform):
        stats = []
        RI_spec = self.get_RI_spec(waveform)
        CNN_outputs = self.apply_random_CNN(RI_spec)
        for out in CNN_outputs:
            stats.append(self.calc_cov(out))
        return stats

    def forward(self, waveform):
        stats = self.get_gram_matrix(waveform)
        losses  = []
        for i in range(len(stats)):
            losses.append(torch.sum((stats[i]-self.ref_stats[i])**2) / torch.sum(self.ref_stats[i]**2)) 
        return 1e9*torch.sum(torch.stack(losses))/len(losses)