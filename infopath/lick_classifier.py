import torch
from tqdm import tqdm
import numpy as np


class Lick_Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_neurons) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_neurons)
        self.fc2 = torch.nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = torch.nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        x = torch.nn.ReLU()(self.fc1(x))
        x = torch.nn.ReLU()(self.fc2(x))
        log_rate = self.fc3(x)
        return log_rate.squeeze(-1)


def prepare_classifier(
    filt_jaw_train,
    filt_jaw_test,
    session_info_train,
    session_info_test,
    lick_counts_train,
    lick_counts_test,
    device,
    remove_mean=False,
    response_time=None,
):
    if response_time is None:
        response_time = filt_jaw_train.shape[0]
    model = Lick_Classifier(response_time, 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    torch.manual_seed(0)
    model.to(device)
    num_session = filt_jaw_train.shape[2]
    poisson_loss = torch.nn.PoissonNLLLoss(log_input=True, full=True)

    pbar = tqdm(range(150))
    for epoch in pbar:
        torch.manual_seed(0)
        test_mse = 0
        for session in range(num_session):
            optimizer.zero_grad()
            inputs = filt_jaw_train[-response_time:, :, session].T
            trials = ~torch.isnan(inputs.sum(1))
            inputs = inputs[trials]
            if remove_mean:
                mean = inputs[:, :20].mean()
                std = inputs[:, :20].std()
                inputs = (inputs - mean) / std

            counts = torch.as_tensor(
                lick_counts_train[session], dtype=torch.float32, device=device
            )[: trials.sum()]

            input_test = filt_jaw_test[-response_time:, :, session].T
            trials_t = ~torch.isnan(input_test.sum(1))
            input_test = input_test[trials_t]
            if remove_mean:
                input_test = (input_test - mean) / std
            counts_test = torch.as_tensor(
                lick_counts_test[session], dtype=torch.float32, device=device
            )[: trials_t.sum()]

            log_rate = model(inputs)
            loss = poisson_loss(log_rate, counts)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_test = torch.exp(model(input_test))
                test_mse += ((pred_test - counts_test) ** 2).mean()

        pbar.set_postfix_str(f"test MSE {test_mse.item()/num_session:.3f}")

    return model
