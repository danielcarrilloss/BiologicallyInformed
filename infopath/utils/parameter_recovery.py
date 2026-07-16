import torch
from infopath.model_loader import load_model_and_optimizer
from infopath.config import load_training_opt, save_opt
import os
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt


def generate_and_save_data(
    save_path="datasets/ModelData",
    log_path="log_dir/a2c6ad1709e4cdb13e66119cc295059401535b3b/2024_1_16_6_56_5",
    trials=400,
    area_readout=1,
    session_name="TS095_20171003",
    save=True,
    seed=0,
    last_best="last",
):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, session_name)):
        os.mkdir(os.path.join(save_path, session_name))
    opt = load_training_opt(log_path)
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    save_opt(save_path, opt)
    model = load_model_and_optimizer(opt, reload=True, last_best=last_best)[0]
    model.rsnn.temperature.data = torch.tensor(model.opt.temperature)
    model.to(model.opt.device)

    stims = torch.ones(trials) * 4
    torch.manual_seed(seed)
    with torch.no_grad():
        # --- BATCHED GENERATION TO PREVENT OOM ---
        batch_size = 200
        spikes_list = []
        for i in range(0, trials, batch_size):
            batch_stims = stims[i:i+batch_size]
            b_spikes, _, _, _ = model(batch_stims)
            spikes_list.append(b_spikes)

        spikes = torch.cat(spikes_list, dim=1)
        # -----------------------------------------

    if "Mechanism" in save_path or "GoNoGo" in save_path:
        a = 8 * model.timestep
        filt = model.filter_fun2(model.filter_fun1(spikes))
        tt1 = filt[:, :, model.rsnn.area_index == 0].mean(2).max((0))[0] > a
        tt2 = filt[:, :, model.rsnn.area_index == 1].mean(2).max((0))[0] > a
        tt3 = filt[:, :, model.rsnn.area_index == 2].mean(2).max((0))[0] > a
        trial_type = tt2 + tt1 * 2 + tt3 * 4
    else:
        trial_type = (
            spikes[:, :, model.rsnn.area_index == area_readout].mean((0, 2))
            / model.timestep
        ) > 13

    print("Hit rate: ", trial_type.unique(return_counts=True)[1] / trials)
    trial_df = pd.DataFrame(
        columns=pd.read_csv(
            os.path.join(opt.datapath, session_name, "trial_info")
        ).keys()[1:]
    )
    if "Mechanism" in save_path or "GoNoGo" in save_path:
        tt = {i: i for i in range(trial_type.max() + 1)}
    else:
        tt = {0: "Miss", 1: "Hit"}
    df_entry = pd.DataFrame(columns=trial_df.columns)
    for i in range(trials):
        df_entry["trial_number"] = [i]
        df_entry["trial_type"] = [tt[trial_type.cpu().numpy()[i]]]
        df_entry["trial_onset"] = [0.1 + i * model.T * model.timestep]
        df_entry["reaction_time_piezo"] = [0.1]
        df_entry["reaction_time_jaw"] = [0.1]
        df_entry["stim"] = [int(stims.numpy()[i])]
        df_entry["trial_active"] = [np.random.randint(0, 2)]
        df_entry["video_onset"] = [2]
        df_entry["video_offset"] = [0]
        trial_df = pd.concat([trial_df, df_entry], ignore_index=True)
    area_dict = {i: area_name for i, area_name in enumerate(opt.areas)}
    if save:
        trial_df.to_csv(os.path.join(save_path, session_name, "trial_info"))
        neuron_df = pd.DataFrame(
            columns=["session", "area", "excitatory", "firing_rate", "cluster_index"]
        )
        neuron_entry = pd.DataFrame(
            columns=["session", "area", "excitatory", "firing_rate", "cluster_index"]
        )
        # dense and sparse implementation
        if True:  # spikes.unique().shape[0] == 2:
            neurons, tms = torch.where(
                spikes.cpu().permute(2, 1, 0).reshape(opt.n_units, -1)
            )
            for i in range(opt.n_units):
                n_ind = int(model.neuron_index[i].item())
                neuron_entry["session"] = [session_name]
                neuron_entry["area"] = [area_dict[model.rsnn.area_index[i].item()]]
                neuron_entry["excitatory"] = [model.rsnn.excitatory_index[i].item()]
                sptms = tms[neurons == i].numpy() * model.timestep
                # baseline firing rate
                neuron_entry["firing_rate"] = [
                    ((sptms % (model.T * model.timestep)) < 0.1).sum() / (trials * 0.1)
                ]
                np.save(
                    os.path.join(
                        save_path, session_name, "neuron_index_{}".format(n_ind)
                    ),
                    sptms,
                )
                neuron_entry["cluster_index"] = [n_ind]
                neuron_df = pd.concat([neuron_df, neuron_entry], ignore_index=True)
        else:
            for i in range(opt.n_units):
                n_ind = int(model.neuron_index[i].item())
                neuron_entry["session"] = [session_name]
                neuron_entry["area"] = [area_dict[model.rsnn.area_index[n_ind].item()]]
                neuron_entry["excitatory"] = [model.rsnn.excitatory_index[n_ind].item()]
                neuron_entry["firing_rate"] = [
                    (spikes[:, :, i].mean() / model.timestep).item()
                ]
                neuron_entry["cluster_index"] = [n_ind]
                neuron_df = pd.concat([neuron_df, neuron_entry], ignore_index=True)
                np.save(
                    os.path.join(
                        save_path, session_name, "neuron_index_{}".format(n_ind)
                    ),
                    spikes[:, :, i].T.reshape(-1).cpu().numpy(),
                )

        neuron_df = neuron_df.sort_values(by=["cluster_index"])
        neuron_df = neuron_df.reset_index(drop=True)
        neuron_df.to_csv(os.path.join(save_path, "cluster_information"))

if __name__ == "__main__":
    generate_and_save_data(
        "datasets/Real_teacher_data",
        "log_dir/main/2026_6_5_6_21_35_full",
        area_readout=1,
        save=True,
        seed=1,
        trials=2000,
        last_best="best",
    )
