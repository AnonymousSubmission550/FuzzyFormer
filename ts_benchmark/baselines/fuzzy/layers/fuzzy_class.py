import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, sqrt,log
from ts_benchmark.baselines.RevIN import RevIN
from ts_benchmark.baselines.time_series_library.layers.Autoformer_EncDec import series_decomp
from numpy import inf, clip
import numpy as np

eps = torch.finfo(torch.float32).tiny


class PatchData(nn.Module):
    def __init__(self, patch_len, padding):
        super(PatchData, self).__init__()
        self.patch_len = patch_len  # 16 P
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = x.permute(0, 2, 1)
        return x


class FuzzyClass(nn.Module):
    def __init__(self, input_dim, output_dim, rule_num, input_channel, delta_0, tau_0, varpi_3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rule_num = rule_num
        self.input_channel = input_channel
        self.dropout = nn.Dropout(p=delta_0)
        self.tau = tau_0 * (1 / (rule_num + eps)) ** (1 / input_dim)
        self.con_type = 1
        self.varpi_3 = varpi_3
        center = torch.empty([self.rule_num, self.input_dim])
        nn.init.xavier_normal_(center)
        self.center = nn.Parameter(center)
        sigma = torch.full([rule_num, input_dim], sqrt(-input_dim/(2*(rule_num+1)*(rule_num+1)*log(0.5))))
        self.sigma = nn.Parameter(sigma)
        if self.con_type == 0:
            self.consequent = self.ZerosConse
            self.consequent_param = nn.Parameter(torch.empty([self.rule_num, self.output_dim]))
            nn.init.constant_(self.consequent_param, 1 / self.input_dim)
        else:
            self.consequent = nn.Linear(self.input_dim, self.rule_num * self.output_dim)
            nn.init.constant_(self.consequent.weight, 1 / self.input_dim)
            nn.init.constant_(self.consequent.bias, 1 / self.input_dim)

    def ZerosConse(self, input_data):
        expanded_matrix = self.consequent_param.unsqueeze(0).unsqueeze(0)
        res = expanded_matrix.repeat(input_data.shape[0], input_data.shape[1], 1, 1)
        return res

    def cv_squared(self, x):
        eps = 1e-20
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, input_data):
        input_data = input_data.permute(0, 2, 1)
        dist = (input_data.unsqueeze(2) - self.center) / (self.sigma + eps)
        fire_strength = torch.exp((-1 / 2 * 1 / self.input_dim * dist.pow(2).sum(dim=-1))) + eps
        norm_fire_strength = fire_strength / (fire_strength.sum(dim=-1, keepdim=True) + eps)
        with torch.no_grad():
            sorted_values, _ = torch.sort(norm_fire_strength, descending=True, dim=-1)
            cumulative_sum = sorted_values.cpu().cumsum(dim=-1).to(sorted_values.device)
            error = cumulative_sum - self.tau
            error[error < 0] = 1
            _, min_indices = torch.min(error, dim=-1)
            values = torch.gather(sorted_values, dim=-1, index=min_indices.unsqueeze(-1))
            mask = (norm_fire_strength >= values).float()
        norm_fire_strength *= mask
        norm_fire_strength = norm_fire_strength / (norm_fire_strength.sum(dim=-1, keepdim=True) + eps)
        Importance = norm_fire_strength.reshape(-1, self.rule_num).sum(0)
        loss_important = self.varpi_3 * self.cv_squared(Importance)
        consequent_vaule = self.consequent(input_data)
        consequent_vaule = consequent_vaule.reshape(-1, self.input_channel, self.rule_num, self.output_dim)
        consequent_vaule = self.dropout(consequent_vaule)
        prediction = torch.einsum("abc, abcd->abd", norm_fire_strength, consequent_vaule).permute(0, 2, 1)
        return prediction[:, -self.output_dim :, :], loss_important


class FuzzyAttentionBase(nn.Module):
    def __init__(self, input_dim, rule_num, delta_0, tau_0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.rule_num = rule_num
        sigma = torch.full([rule_num, input_dim], sqrt(-input_dim/(2*(rule_num+1)*(rule_num+1)*log(0.5))))
        self.sigma = nn.Parameter(sigma)
        self.dropout = nn.Dropout(p=delta_0)
        self.tau = tau_0 * (1 / (rule_num + eps)) ** (1 / input_dim)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, queries, keys, values):
        dist = (queries - keys) / (self.sigma + eps)
        fire_strength = torch.exp((-1 / 2 * 1 / self.input_dim * dist.pow(2).sum(dim=-1))) + eps
        norm_fire_strength = fire_strength / (fire_strength.sum(dim=-1, keepdim=True) + eps)
        with torch.no_grad():
            sorted_values, _ = torch.sort(norm_fire_strength, descending=True, dim=-1)
            cumulative_sum = sorted_values.cpu().cumsum(dim=-1).to(sorted_values.device)
            error = cumulative_sum - self.tau
            error = torch.where(error < 0, torch.ones_like(error), error)
            _, min_indices = torch.min(error, dim=-1)
            norm_fire_values = torch.gather(sorted_values, dim=-1, index=min_indices.unsqueeze(-1))
            mask = (norm_fire_strength >= norm_fire_values).float()
        norm_fire_strength = norm_fire_strength * mask
        norm_fire_strength = norm_fire_strength / (norm_fire_strength.sum(dim=-1, keepdim=True) + eps)
        attention_consequent = self.dropout(values)
        prediction = torch.einsum("abc, abcd->abd", norm_fire_strength, attention_consequent).permute(0, 2, 1)
        Importance = norm_fire_strength.reshape(-1, self.rule_num).sum(0)
        loss_important = self.cv_squared(Importance)
        return prediction, loss_important
    
class FuzzyAttention(nn.Module):
    def __init__(self, patch_size, rule_num, input_channel, model_size, varpi_1, varpi_2, delta_0, tau_0):
        super().__init__()
        self.input_dim = patch_size
        self.rule_num = rule_num
        self.model_size = model_size
        self.input_channel = input_channel
        self.query_projection = nn.Linear(patch_size, model_size)
        self.key_projection = nn.Linear(patch_size, model_size * rule_num)
        self.value_projection = nn.Linear(patch_size, model_size * rule_num)
        self.out_projection = nn.Linear(model_size, patch_size)
        self.inner_attention = FuzzyAttentionBase(model_size, rule_num, delta_0, tau_0)
        self.varpi_1 = varpi_1
        self.varpi_2 = varpi_2
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.xavier_uniform_(self.out_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.zeros_(self.key_projection.bias)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.zeros_(self.out_projection.bias)
        
    def constraint_loss(self):
        W1 = self.query_projection.weight
        W2 = self.out_projection.weight
        identity_matrix = torch.eye(W1.size(0)).to(W1.device)
        weight_loss = torch.norm(torch.matmul(W1, W2) - identity_matrix, p="fro")
        b1 = self.query_projection.bias
        b2 = self.out_projection.bias
        bias_loss = torch.norm(b1, p=2) + torch.norm(b2, p=2)
        key_l1_loss = torch.norm(self.key_projection.weight, p=1) + torch.norm(self.key_projection.bias, p=1)
        value_l1_loss = torch.norm(self.value_projection.weight, p=1) + torch.norm(self.value_projection.bias, p=1)
        loss1 = self.varpi_1 * (weight_loss + bias_loss)
        loss2 = self.varpi_2 * (key_l1_loss + value_l1_loss)
        return loss1 + loss2

    def forward(self, input_data):
        input_data = input_data.permute(0, 2, 1)
        queries = self.query_projection(input_data).unsqueeze(2)
        keys = self.key_projection(input_data)
        keys = keys.reshape(-1, self.input_channel, self.rule_num, self.model_size)
        values = self.value_projection(input_data)
        values = values.reshape(-1, self.input_channel, self.rule_num, self.model_size)
        fuzzy_attention_vaule, loss_important = self.inner_attention(queries, keys, values)
        attention_vaule = self.out_projection(fuzzy_attention_vaule.permute(0, 2, 1)).permute(0, 2, 1)
        loss_attention = self.constraint_loss()
        return attention_vaule, loss_attention


def Adjust_shape(x_enc, batch_size_now):
    x_enc = x_enc.reshape((x_enc.shape[0], -1))
    x_enc = x_enc.reshape((batch_size_now, -1, x_enc.shape[-1]))
    x_enc = x_enc.permute(0, 2, 1)
    return x_enc
