import torch
import torch.nn as nn

class ConsecutiveDifferenceHigherOrderLossBatch(nn.Module):
    def __init__(self, consecutive_size, order=1):
        super(ConsecutiveDifferenceHigherOrderLossBatch, self).__init__()
        self.consecutive_size = consecutive_size
        self.order = order

    def forward(self, prediction, target):
        pred_reshape = prediction.view(-1, self.consecutive_size)
        target_reshape = target.view(-1, self.consecutive_size)
        result = torch.zeros(1, device=prediction.device)  # To ensure it uses the same device as the input


        pred_a = pred_reshape[1:, :]
        pred_b = pred_reshape[:-1, :]
        target_a = target_reshape[1:, :]
        target_b = target_reshape[:-1, :]
        for i in range(self.order):
            pred_dif = pred_a - pred_b
            target_dif = target_a - target_b
            pred_a = pred_dif[1:, :]
            pred_b = pred_dif[:-1, :]
            target_a = target_dif[1:, :]
            target_b = target_dif[:-1, :]

            result += torch.mean((pred_dif - target_dif) ** 2) / self.order
        return result


class ConsecutiveDifferenceHigherOrderLoss(nn.Module):
    def __init__(self, consecutive_size, order=1):
        super(ConsecutiveDifferenceHigherOrderLoss, self).__init__()
        self.consecutive_size = consecutive_size
        self.order = order

    def forward(self, prediction, target):
        pred_reshape = prediction.view(-1, self.consecutive_size)
        target_reshape = target.view(-1, self.consecutive_size)
        result = torch.zeros(1, device=prediction.device)  # To ensure it uses the same device as the input


        pred_a = pred_reshape[:, 1:]
        pred_b = pred_reshape[:, :-1]
        target_a = target_reshape[:, 1:]
        target_b = target_reshape[:, :-1]
        for i in range(self.order):
            pred_dif = pred_a - pred_b
            target_dif = target_a - target_b
            pred_a = pred_dif[:, 1:]
            pred_b = pred_dif[:, :-1]
            target_a = target_dif[:, 1:]
            target_b = target_dif[:, :-1]

            result += torch.mean((pred_dif - target_dif) ** 2) / self.order
        return result

