import torch
import torch.nn as nn
import numpy as np


class InputSpikes(nn.Module):
    def __init__(self, opt, prec_type=torch.float32):
        super(InputSpikes, self).__init__()
        self.prec_type = prec_type
        self.opt = opt
        if opt.n_rnn_in == 200:
            p_stim = 0.5
        elif opt.n_rnn_in == 300:
            p_stim = 2 / 3
        else:
            p_stim = 1
        self.p_stim = p_stim

    def forward(self, stim):
        dt_in_seconds = self.opt.dt / 1000.0
        n_neurons = self.opt.n_rnn_in
        start = -int(self.opt.start / dt_in_seconds)
        stop = int(self.opt.stop / dt_in_seconds)
        thalamic_delay = int(self.opt.thalamic_delay / dt_in_seconds)
        stim_dur = int(self.opt.stim_duration / dt_in_seconds)
        firing_prob = self.opt.input_f0 * dt_in_seconds
        batch_size = stim.shape[0]
        pattern = (
            torch.ones((start + stop, batch_size, n_neurons), device=stim.device)
            * firing_prob
        )
        pattern[pattern > 1] = 1
        n_stim = int(n_neurons * self.p_stim / len(self.opt.stim_onsets))
        for i, j in enumerate(stim):
            for l, k in enumerate(self.opt.stim_onsets):
                start = int(np.round((k - self.opt.start) / dt_in_seconds))
                scale = (
                    self.opt.scale_fun(j.item()) if k == 0 else self.opt.scale_fun(1)
                )
                pattern[
                    start + thalamic_delay : start + thalamic_delay + stim_dur,
                    i,
                    n_stim * l : n_stim * (l + 1),
                ] *= (
                    1 + self.opt.stim_valance * scale
                )
        pattern[pattern > 1] = 1
        if n_neurons == len(self.opt.stim_onsets):
            return (pattern > firing_prob) * 1.0 * self.opt.dt / 2
        return torch.bernoulli(pattern)


class InputSpikes_Adapted(nn.Module):
    def __init__(self, opt, prec_type=torch.float32):
        super().__init__()
        self.prec_type = prec_type
        self.opt = opt
        self.tau_sound_decay = getattr(opt, "tau_sound_decay", 0.5)
        if opt.n_rnn_in == 600:
            p_stim = 0.5
        elif opt.n_rnn_in == 300:
            p_stim = 2 / 3
        else:
            p_stim = 1
        self.p_stim = p_stim


    def forward(self, stim):
        opt = self.opt
        dt_in_seconds = opt.dt / 1000.0
        n_neurons = opt.n_rnn_in
        n_per_group = n_neurons // 3 
        n_active = int(n_per_group * self.p_stim)
        
        start = -int(opt.start / dt_in_seconds)
        stop = int(opt.stop / dt_in_seconds)
        sound_dur = int(opt.stim_duration / dt_in_seconds)
        thalamic_delay = int(opt.thalamic_delay / dt_in_seconds)
        tau_bins = self.tau_sound_decay / dt_in_seconds
        firing_prob = opt.input_f0 * dt_in_seconds
        batch_size = stim.shape[0]

        pattern = torch.ones((start + stop, batch_size, n_neurons), device=stim.device) * firing_prob
        pattern[pattern > 1] = 1

        decay_kernel = torch.exp(-torch.arange(sound_dur, device=stim.device, dtype=self.prec_type) / tau_bins)
        decay_expanded = decay_kernel[:sound_dur].unsqueeze(1) # Shape: (dur, 1)

        k = opt.stim_onsets[0]
        onset = int(np.round((k - opt.start) / dt_in_seconds))
        t0 = onset + thalamic_delay
        t1 = min(t0 + sound_dur, pattern.shape[0])
        dur = t1 - t0
        
        if dur <= 0:
            return torch.bernoulli(pattern)

        decay_expanded = decay_expanded[:dur]

        for i, j in enumerate(stim):
            is_go = j.item() > 0.5      # Trial type
            scale = opt.scale_fun(1) 
            valance = opt.stim_valance * scale

            # 1. Constant channel
            pattern[t0:t1, i, 0 : n_active] *= (1 + valance)

            # 2. Active Channel (GO / NO-GO):
            if is_go:
                pattern[t0:t1, i, n_per_group : n_per_group + n_active] *= (1 + valance * decay_expanded)
            else:
                pattern[t0:t1, i, 2 * n_per_group : 2 * n_per_group + n_active] *= (1 + valance * decay_expanded)

        pattern[pattern > 1] = 1
        return torch.bernoulli(pattern)
