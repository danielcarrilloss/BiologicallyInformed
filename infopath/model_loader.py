from infopath.session_stitching import build_network
from models.pop_rsnn import PopRSNN
from datasets.prepare_input import InputSpikes
from datasets.prepare_input import InputSpikes_Adapted
import numpy as np
import torch.nn as nn
import torch
from infopath.utils.logger import optimizer_to, reload_weights
import os
import copy
from geomloss import SamplesLoss
from infopath.config import load_training_opt
import random
from torchmultitask.splitter import NormalizedMultiTaskSplitter
from infopath.losses import *


class FullModel(nn.Module):
    def __init__(self, opt):
        super(FullModel, self).__init__()
        self.opt = opt
        self.num_areas = opt.num_areas
        self.rsnn = init_rsnn(opt)
        #self.input_spikes = InputSpikes(opt)
        self.input_spikes = InputSpikes_Adapted(opt)
        start, stop = self.opt.start, self.opt.stop
        self.opt.start, self.opt.stop = 0.0, 0.1
        #self.input_spikes_pre = InputSpikes(copy.deepcopy(opt))
        self.input_spikes_pre = InputSpikes_Adapted(copy.deepcopy(opt))
        self.opt.start, self.opt.stop = start, stop
        self.timestep = self.opt.dt * 0.001
        self.trial_onset = -int(self.opt.start / self.timestep)
        # some filter functions
        kernel_size1 = int(self.opt.psth_filter / self.opt.dt)
        padding = int((kernel_size1 - 1) / 2)
        self.filter1 = torch.nn.AvgPool1d(
            kernel_size1,
            int(kernel_size1 // 2),
            padding=padding,
            count_include_pad=False,
        )

        stride2 = opt.pop_avg_resample
        padding = int((stride2 * 2 - 1) / 2)
        self.filter2 = torch.nn.AvgPool1d(
            2 * stride2, stride2, padding=padding, count_include_pad=False
        )

        self.T = int((self.opt.stop - self.opt.start) / self.opt.dt * 1000)
        input_dimD = int(np.round(self.T / stride2 / int(kernel_size1 // 2)))
        self.input_dimD = input_dimD
        self.jaw_mean = torch.nn.Parameter(torch.zeros(1))
        self.jaw_std = torch.nn.Parameter(torch.ones(1) * 0.1)

        # for the trial-matching loss function
        if opt.geometric_loss:
            self.trial_loss_fun = SamplesLoss(loss="sinkhorn", p=1, blur=0.01)
        else:
            self.trial_loss_fun = hard_trial_matching_loss
        # setup the loss splitter
        tasks = {}
        if opt.loss_trial_wise:
            tasks["trial_loss"] = 2
        if opt.loss_neuron_wise:
            tasks["neuron_loss"] = 1
        if opt.loss_cross_corr:
            tasks["cross_corr_loss"] = 1
        if opt.loss_firing_rate:
            tasks["firing_rate_loss"] = 1
        if len(tasks) <= 1:
            self.opt.with_task_splitter = False
        self.multi_task_splitter = NormalizedMultiTaskSplitter(tasks)

    def filter_fun1(self, spikes):
        """filter spikes

        Args:
            spikes (torch.tensor): spikes with dimensionality time x trials x neurons
        Return:
            filtered spikes with kernel specified from the self.opt
        """
        if spikes is None:
            return None
        spikes = spikes.permute(2, 1, 0)
        spikes = self.filter1(spikes)
        return spikes.permute(2, 1, 0)

    def filter_fun2(self, x):
        """filter signal

        Args:
            x (torch.tensor): signal with dimensionality time x trials x neurons
        Return:
            filtered signals with kernel specified from the self.opt
        """
        if x is None:
            return None
        x = x.permute(2, 1, 0)
        x = self.filter2(x)
        return x.permute(2, 1, 0)

    @torch.no_grad()
    def steady_state(self, state=None, sample_trial_noise=True):
        """
        set state of network in the steady state (basically run the network for 300ms
        from a zero state) we don't train this part so torch.no_grad
        """
        trials = torch.zeros(self.opt.batch_size).long().to(self.opt.device)
        spike_data = self.input_spikes_pre(trials)
        if state is None:
            state = self.rsnn.zero_state(self.opt.batch_size)
        if sample_trial_noise:
            self.rsnn.sample_trial_noise(self.opt.batch_size)
        if self.opt.lsnn_version != "mlp":
            self.rsnn.sample_mem_noise(spike_data.shape[0], self.opt.batch_size)
            _, _, _, state = self.rsnn(spike_data, state)
        return state

    def step(
        self,
        input_spikes,
        state,
        mem_noise,
        start=None,
        stop=None,
        light=None,
        data=None,
        dt=None,
    ):
        opt = self.opt
        if start is None:
            start = 0
        if stop is None:
            stop = input_spikes.shape[0]
        self.rsnn.mem_noise = mem_noise[start:stop]
        spike_outputs, voltages, model_jaw, state = self.rsnn(
            input_spikes[start:stop].to(opt.device), state, light=light, data=data
        )
        if not opt.scaling_jaw_in_model:
            model_jaw = (model_jaw - self.jaw_mean) / self.jaw_std
            if self.opt.jaw_nonlinear:
                model_jaw = torch.exp(model_jaw) - 1
        return spike_outputs, voltages, model_jaw, state

    def forward(self, stims, step=None, light=None, data=None):
        self.opt.batch_size = stims.shape[0]
        state = self.steady_state()
        input_spikes = self.input_spikes(stims)
        self.rsnn.sample_mem_noise(self.T, stims.shape[0], step)
        mem_noise = self.rsnn.mem_noise.clone()
        return self.step(input_spikes, state, mem_noise, light=light, data=data)

    def step_with_dt(
        self, input_spikes, state, mem_noise, light=None, dt=25
    ):  # not really used in the paper, usefull for running forward the network without training in smaller gpus
        spikes, voltages, jaw, l = [], [], [], None
        for i in range(np.ceil(input_spikes.shape[0] / dt).astype(int)):
            self.rsnn.mem_noise = mem_noise[i * dt : (i + 1) * dt].clone()
            if light is not None:
                l = light[i * dt : (i + 1) * dt]
            sp, v, j, state = self.rsnn(input_spikes[i * dt : (i + 1) * dt], state, l)
            spikes.append(sp)
            voltages.append(v)
            jaw.append(j)
        spikes = torch.cat(spikes, dim=0)
        voltages = torch.cat(voltages, dim=0)
        jaw = torch.cat(jaw, dim=0)
        self.rsnn.mem_noise = mem_noise
        if not self.opt.scaling_jaw_in_model:
            jaw = (jaw - self.jaw_mean) / self.jaw_std
            if self.opt.jaw_nonlinear:
                model_jaw = torch.exp(model_jaw) - 1
        return spikes, voltages, jaw, state

    # Where we add all the terms of the loss function
    def generator_loss(
        self,
        model_spikes,
        data_spikes,
        model_jaw,
        data_jaw,
        session_info,
        netD,
    ):
        """Calculates the generator loss (in the no GAN like cases is the only/classic loss), if there are multiple loss
        they are weighted with the loss spliter to ensure that their gradients have comparable scales.

        Args:
            model_spikes (torch.tensor): The spikes from the model $z$
            data_spikes (torch.tensor): The spikes from the recordings $z^{\mathcal{D}}$
            model_jaw (torch.tensor, None): The jaw trace from the model
            data_jaw (torch.tensor, None): The jaw trace from the data
            session_info (list): List with information about the different sessions
            netD (torch.nn.Module, None): _description_

        Returns:
            torch.float: Total loss
        """
        count_tasks = 0  # useful for the the loss_splitter
        with_jaw = -1 if len(self.opt.motor_areas) > 0 else None
        if self.opt.with_task_splitter:
            if len(self.opt.motor_areas) > 0:
                model_output = self.multi_task_splitter(
                    torch.cat([model_spikes, model_jaw], dim=2)
                )
            else:
                model_output = self.multi_task_splitter(model_spikes)
        else:
            if len(self.opt.motor_areas) > 0:
                model_output = [torch.cat([model_spikes, model_jaw], dim=2)]
            else:
                model_output = [model_spikes]
        opt = self.opt
        # filter once for the $T_{neuron}$ and twice for the $T_{trial}$
        filt_model = self.filter_fun1(model_spikes)
        filt_data = self.filter_fun1(data_spikes)
        if data_jaw is not None:
            filt_data_jaw = self.filter_fun1(data_jaw)
        else:
            filt_data_jaw = None
        # Generator loss
        neur_loss, trial_loss, fr_loss, cc_loss, tm_mle_loss = 0, 0, 0, 0, 0
        if opt.loss_firing_rate:  # not used in the paper
            if self.opt.with_task_splitter:
                fr_loss += firing_rate_loss(
                    data_spikes,
                    model_output["firing_rate_loss"][..., :with_jaw],
                    self.trial_onset,
                    self.timestep,
                )
            else:
                fr_loss += firing_rate_loss(
                    data_spikes, model_spikes, self.trial_onset, self.timestep
                )
        if opt.loss_cross_corr:  # not used in the paper
            if self.opt.with_task_splitter:
                cc_loss = cross_corr_loss(
                    self,
                    data_spikes,
                    model_output["cross_corr_loss"][..., :with_jaw],
                    session_info,
                )[0]
                # if self.opt.stats_loss:
                #     cc_loss += cc_var
                count_tasks += 1
            else:
                cc_loss = cross_corr_loss(
                    self, data_spikes, model_spikes, session_info
                )[0]
        if opt.loss_neuron_wise:  # T_{neuron}
            if self.opt.with_task_splitter:
                filt_model = self.filter_fun1(
                    model_output["neuron_loss"][..., :with_jaw]
                )
                filt_jaw = self.filter_fun1(model_output["neuron_loss"][..., with_jaw:])
            f_data = filt_data[:, :, self.neuron_index != -1]
            f_model = filt_model[:, :, self.neuron_index != -1]
            psth_model, psth_data, f_model_norm = z_score_norm(f_data, f_model)
            if self.opt.stats_loss:
                correction_term = f_model_norm.var(1).mean() / f_model_norm.shape[1]
                neur_loss += ((psth_model - psth_data) ** 2).mean() - correction_term
            else:
                neur_loss += ((psth_model - psth_data) ** 2).mean()
            if opt.with_behaviour and not opt.loss_trial_wise:
                f_data_jaw = filt_data_jaw
                f_data_jaw = torch.cat(torch.unbind(f_data_jaw, dim=2), dim=1)
                f_data_jaw = f_data_jaw[:, ~torch.isnan(f_data_jaw.sum(0))]
                psth_model, psth_data, f_model_norm = z_score_norm(
                    f_data_jaw, filt_jaw[:, :, 0]
                )
                neur_loss += ((psth_model - psth_data) ** 2).mean()
        if opt.loss_trial_wise:  # T_{trial}
            if opt.gan_loss:  #  if True for GAN else trial matching
                if self.opt.with_task_splitter:
                    model_spikes = model_output["trial_loss"][..., :with_jaw]
                    model_jaw = model_output["trial_loss"][..., with_jaw:]
                # for the GAN loss usually is good to deactivate the loss_splitter
                if self.opt.t_trial_gan:
                    trial_loss, _ = discriminator_loss(
                        netD,
                        self.filter_fun2(self.filter_fun1(model_spikes)),
                        self.filter_fun2(filt_data),
                        self.filter_fun2(filt_data_jaw),
                        self.filter_fun2(self.filter_fun1(model_jaw)),
                        session_info,
                        self.rsnn.area_index,
                        self.rsnn.excitatory_index,
                        discriminator=False,
                        t_trial_gan=self.opt.t_trial_gan,
                        z_score=self.opt.z_score,
                    )
                else:
                    trial_loss, _ = discriminator_loss(
                        netD,
                        model_spikes,
                        data_spikes,
                        data_jaw,
                        model_jaw,
                        session_info,
                        self.rsnn.area_index,
                        self.rsnn.excitatory_index,
                        discriminator=False,
                        t_trial_gan=self.opt.t_trial_gan,
                        z_score=self.opt.z_score,
                    )
            else:
                if self.opt.with_task_splitter:
                    model_spikes = model_output["trial_loss"][..., :with_jaw]
                    model_jaw = model_output["trial_loss"][..., with_jaw:]
                trial_loss = trial_matching_loss(
                    self,
                    self.filter_fun2(filt_data),
                    self.filter_fun2(self.filter_fun1(model_spikes)),
                    session_info,
                    self.filter_fun2(filt_data_jaw),
                    self.filter_fun2(self.filter_fun1(model_jaw)),
                    self.trial_loss_fun,
                    self.rsnn.area_index,
                    self.rsnn.excitatory_index,
                    z_score=self.opt.z_score,
                    trial_loss_area_specific=self.opt.trial_loss_area_specific,
                    trial_loss_exc_specific=self.opt.trial_loss_exc_specific,
                    feat_svd=False,
                )
                if self.opt.feat_svd:
                    trial_loss += (
                        trial_matching_loss(
                            self,
                            self.filter_fun2(filt_data),
                            self.filter_fun2(self.filter_fun1(model_spikes)),
                            session_info,
                            self.filter_fun2(filt_data_jaw),
                            self.filter_fun2(self.filter_fun1(model_jaw)),
                            self.trial_loss_fun,
                            self.rsnn.area_index,
                            self.rsnn.excitatory_index,
                            z_score=self.opt.z_score,
                            trial_loss_area_specific=self.opt.trial_loss_area_specific,
                            trial_loss_exc_specific=self.opt.trial_loss_exc_specific,
                            feat_svd=True,
                            dim=10,
                        )
                        * 100
                    )
        if opt.loss_trial_matched_mle:
            if self.opt.with_task_splitter:
                filt_model = self.filter_fun1(model_output[count_tasks][..., :with_jaw])
            tm_mle_loss += trial_matched_mle_loss(
                self.filter_fun2(filt_data),
                self.filter_fun2(filt_model),
                session_info,
                data_jaw,
                model_jaw,
            )
        fr_loss = opt.coeff_fr_loss * fr_loss
        trial_loss *= opt.coeff_trial_loss
        neur_loss *= opt.coeff_loss
        cc_loss *= opt.coeff_cross_corr_loss
        tm_mle_loss *= opt.coeff_loss
        return fr_loss, trial_loss, neur_loss, cc_loss, tm_mle_loss

    def mean_activity(self, activity, clusters=None):
        with torch.no_grad():
            device = self.opt.device
            activity = self.filter_fun1(activity.to(device)).cpu()
            if clusters is None:
                clusters = torch.arange(activity.shape[2]) > 0
            step = self.timestep
            activity = activity[..., clusters]
            exc_index = self.rsnn.excitatory_index[clusters]
            area = self.rsnn.area_index[clusters]
            mean_exc, mean_inh = [], []
            for i in range(self.num_areas):
                area_index = area == i
                exc_mask = exc_index & area_index
                exc_mask = exc_mask.cpu()
                simulation_exc = (
                    np.nanmean(activity[..., exc_mask].cpu(), (1, 2)) / step
                )
                mean_exc.append(simulation_exc)
                inh_index = ~exc_index
                inh_mask = inh_index & area_index
                inh_mask = inh_mask.cpu()
                simulation_inh = (
                    np.nanmean(activity[..., inh_mask].cpu(), (1, 2)) / step
                )
                mean_inh.append(simulation_inh)
        return mean_exc, mean_inh


def load_model_and_optimizer(opt, reload=False, last_best="last", reload_optim=True):
    model = FullModel(opt)
    if "opt.json" in os.listdir(opt.datapath):
        opt_temp = load_training_opt(opt.datapath)
        log_path_temp = opt_temp.log_path
    else:
        log_path_temp = opt.log_path
    if os.path.exists(os.path.join(log_path_temp, "sessions.npy")):
        sessions = np.load(
            os.path.join(log_path_temp, "sessions.npy"), allow_pickle=True
        )
        model.neuron_index = np.load(os.path.join(log_path_temp, "neuron_index.npy"))
        model.firing_rate = np.load(os.path.join(log_path_temp, "firing_rate.npy"))
        model.areas = np.load(
            os.path.join(log_path_temp, "areas.npy"), allow_pickle=True
        )
        model.sessions = sessions
    else:
        np.random.seed(opt.seed)
        neuron_index, firing_rate, session, area = build_network(
            model.rsnn, opt.datapath, opt.areas, opt.with_behaviour, opt.hidden_perc
        )
        model.neuron_index = neuron_index
        model.firing_rate = firing_rate
        model.areas = area
        model.sessions = session

    if opt.gan_loss:
        if opt.t_trial_gan:  # for spikeT-GAN
            netD = DiscriminatorSession(
                model, opt.gan_hidden_neurons, opt.with_behaviour
            )
        else:  # for spike-GAN
            netD = DiscriminatorSessionCNN(
                model, opt.gan_hidden_neurons, opt.with_behaviour
            )
        optimizerD = torch.optim.AdamW(netD.parameters(), lr=opt.lr, weight_decay=0.0)
        netD.to(opt.device)
    else:
        netD, optimizerD = None, None

    # if not reload and ("opt.json" in os.listdir(opt.datapath)):
    #     opt_temp = load_training_opt(opt.datapath)
    #     model_temp = FullModel(opt_temp)
    #     reload_weights(opt_temp, model_temp, last_best="best")
    #     delattr(model.rsnn, "_w_in")
    #     model.rsnn.register_buffer("_w_in", model_temp.rsnn._w_in.data)
    #     opt.train_input_weights = False
    #     if opt.spike_function == opt_temp.spike_function:
    #         delattr(model.rsnn, "v_rest")
    #         model.rsnn.register_buffer("v_rest", model_temp.rsnn.v_rest.data)
    #         opt.train_bias = False
    #         delattr(model.rsnn, "bias")
    #         model.rsnn.register_buffer("bias", model_temp.rsnn.bias.data)
    #         opt.train_noise_bias = False
    #     del model_temp

    optimizerG = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=opt.w_decay
    )
    if reload:
        reload_weights(
            opt, model, optimizerG, last_best=last_best, reload_optim=reload_optim
        )
        if opt.gan_loss:
            optim_path = os.path.join(opt.log_path, "last_optimD.ckpt")
            netD_path = os.path.join(opt.log_path, "last_netD.ckpt")
            optimizerD.load_state_dict(
                torch.load(optim_path, map_location=opt.device.type)
            )
            netD.load_state_dict(torch.load(netD_path, map_location=opt.device.type))

    model.to(opt.device)
    optimizer_to(optimizerG, opt.device)
    return model, netD, optimizerG, optimizerD


def init_rsnn(opt):
    if (opt.spike_function == "sigmoid") and opt.with_reset:
        from models.rsnn import RSNN
    else:
        from models.rsnn_nocond_nojawfeedback import RSNN

    if opt.lsnn_version == "simplified":
        rsnn = RSNN(
            opt.n_rnn_in,
            opt.n_units,
            sigma_mem_noise=opt.noise_level_list,
            num_areas=opt.num_areas,
            tau_adaptation=opt.tau_adaptation,
            tau=opt.tau_list,
            exc_inh_tau_mem_ratio=opt.exc_inh_tau_mem_ratio,
            n_delay=opt.n_delay,
            inter_delay=opt.inter_delay,
            restrict_inter_area_inh=opt.restrict_inter_area_inh,
            prop_adaptive=opt.prop_adaptive,
            dt=opt.dt,
            p_exc=opt.p_exc,
            spike_function=opt.spike_function,
            train_v_rest=opt.train_bias,
            trial_offset=opt.trial_offset,
            rec_groups=opt.rec_groups,
            latent_space=opt.latent_space,
            train_adaptation=opt.train_adaptation,
            train_noise_bias=opt.train_noise_bias,
            conductance_based=opt.conductance_based,
            jaw_delay=opt.jaw_delay,
            jaw_min_delay=opt.jaw_min_delay,
            tau_jaw=opt.tau_jaw,
            motor_areas=opt.motor_areas,
            temperature=opt.temperature,
            latent_new=opt.latent_new,
            jaw_open_loop=opt.jaw_open_loop,
            scaling_jaw_in_model=opt.scaling_jaw_in_model,
            p_exc_in=opt.p_exc_in,
            v_rest=opt.v_rest,
            thr=opt.thr,
            trial_offset_bound=opt.trial_offset_bound,
            block_graph=opt.block_graph,
            weights_distance_based=opt.weights_distance_based,
            train_input_weights=opt.train_input_weights,
            p_ee=opt.p,
            p_ei=opt.p,
            p_ie=opt.p,
            p_ii=opt.p,
            weights_random_delays=opt.weights_random_delays,
            with_reset=opt.with_reset,
        )
    elif opt.lsnn_version == "mlp":
        T = int((opt.stop - opt.start) / opt.dt * 1000)
        rsnn = MLP(opt.n_rnn_in, opt.n_units, T)
    else:
        rsnn = PopRSNN(
            opt.n_rnn_in,
            opt.n_units,
            sigma_mem_noise=opt.noise_level_list,
            num_areas=opt.num_areas,
            tau=opt.tau_list,
            n_delay=opt.n_delay,
            inter_delay=opt.inter_delay,
            dt=opt.dt,
            p_exc=opt.p_exc,
            rec_groups=opt.rec_groups,
            latent_space=opt.latent_space,
            train_noise_bias=opt.train_noise_bias,
            jaw_delay=opt.jaw_delay,
            jaw_min_delay=opt.jaw_min_delay,
            tau_jaw=opt.tau_jaw,
            motor_areas=opt.motor_areas,
            latent_new=opt.latent_new,
            jaw_open_loop=opt.jaw_open_loop,
            scaling_jaw_in_model=opt.scaling_jaw_in_model,
            p_exc_in=opt.p_exc_in,
            trial_offset=opt.trial_offset,
            pop_per_area=opt.pop_per_area,
        )
    return rsnn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_neurons, T):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim * T + 10 * 10, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, hidden_neurons * T)
        self.area_index = torch.zeros(500)
        self.area_index[200:400] = 1
        self.area_index[450:] = 1
        self.excitatory_index = torch.zeros(500)
        self.excitatory_index[:400] = 1
        self.area_index = self.area_index.long()
        self.excitatory_index = self.excitatory_index > 0
        self.n_units = hidden_neurons
        self._w_rec = self.fc1.weight[None].data.clone()
        self._w_in = self.fc2.weight.data.clone()
        self.motor_areas = torch.tensor([])
        self.excitatory = 400
        self.inhibitory = 100
        rand_projection_neurons = torch.randn(hidden_neurons, 10)
        self.register_buffer("rand_projection_neurons", rand_projection_neurons)
        rand_projection_time = torch.randn(10, T)
        self.register_buffer("rand_projection_time", rand_projection_time)

    def forward(self, x, state, light=None, data=None):
        # x = torch.einsum("tn,nkm->tkm", self.rand_projection_time, x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)
        mem_noise = torch.einsum(
            "tn,nkm->tkm",
            self.rand_projection_time,
            self.mem_noise,
            # self.rand_projection_neurons,
        )
        # mem_noise = mem_noise.clone()
        mem_noise = mem_noise.permute(1, 0, 2)
        mem_noise = mem_noise.reshape(mem_noise.shape[0], -1)
        x = torch.cat([x, mem_noise], 1)
        x = torch.nn.ReLU()(self.fc1(x))
        x = torch.nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        output = x.reshape(x.shape[0], -1, 500)
        output = output.permute(1, 0, 2)
        return output, output.clone(), output[:, :, :2], state

    def reform_recurent(self, a, l1_decay):
        pass

    def reform_v_rest(self):
        pass

    def zero_state(self, x):
        return 0

    @torch.no_grad()
    def sample_mem_noise(self, T, trials, step=None):
        if step is not None:
            seed(step)
        self.mem_noise = torch.randn(T, trials, 10).to(self.fc1.weight.device)

    def sample_trial_noise(self, trials):
        self.trial_noise = torch.randn(trials)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_neurons) -> None:
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, int(hidden_neurons // 2))
        self.fc3 = nn.Linear(int(hidden_neurons // 2), 1)

    def forward(self, x):
        x = torch.nn.LeakyReLU(0.2)(self.fc1(x))
        x = torch.nn.LeakyReLU(0.2)(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class DiscriminatorSession(nn.Module):
    def __init__(self, model, hidden_neurons, with_behaviour=True):
        super(DiscriminatorSession, self).__init__()
        self.discriminators = torch.nn.ModuleList()
        sessions = np.unique(model.sessions)
        input_dim = model.input_dimD
        for session in sessions:
            n_areas = num_areas(model, session) + with_behaviour * 1
            self.discriminators.append(
                Discriminator(input_dim * n_areas, hidden_neurons)
            )


class DiscriminatorCNN(nn.Module):
    def __init__(self, input_dim, hidden_neurons):
        super(DiscriminatorCNN, self).__init__()
        kernel_size = [7, 7, 7]
        stride = [3, 3, 3]
        filters = [input_dim[1], 32, 32]
        time_points = input_dim[0]

        layers = []
        for i in range(len(filters) - 1):
            layers.append(
                nn.Conv1d(
                    filters[i],
                    filters[i],
                    kernel_size[i],
                    stride=stride[i],
                    groups=filters[i],
                )
            )
            layers.append(nn.Conv1d(filters[i], filters[i + 1], 1))
            layers.append(nn.BatchNorm1d(filters[i + 1]))
            time_points = int((time_points - kernel_size[i]) / stride[i] + 1)
            layers.append(nn.ReLU())
        self.input_block = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(time_points * filters[i + 1], hidden_neurons).cuda()
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_neurons, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.dropout = nn.Dropout(0.0)
        self.leaky_relu = nn.ReLU()

    def forward(self, x):
        x = self.input_block(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class DiscriminatorSessionCNN(nn.Module):
    def __init__(self, model, hidden_neurons, with_behaviour=True):
        super(DiscriminatorSessionCNN, self).__init__()
        self.discriminators = torch.nn.ModuleList()
        sessions = np.unique(model.sessions)
        kernel_size1 = int(model.opt.psth_filter / model.opt.dt)
        stride1 = int(kernel_size1 // 2)
        input_dim = model.T
        for session in sessions:
            neurons = (model.sessions == session).sum() + with_behaviour
            self.discriminators.append(
                DiscriminatorCNN([input_dim, neurons], hidden_neurons)
            )


def num_areas(model, session):
    area_index = model.rsnn.area_index
    areas = area_index[session == model.sessions].unique()
    num_areas = len(areas)
    for area in areas:
        if (area_index[session == model.sessions] == area).sum() < 10:
            num_areas -= 1
    return num_areas


def seed(seed=1810):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

