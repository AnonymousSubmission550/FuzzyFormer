from ts_benchmark.baselines.fuzzy.layers.fuzzy_class import *


class FuzzyModel(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        input_channel = configs.enc_in
        patch_len = configs.patch_len
        fsa_rule_num = configs.fsa_rule_num
        fs_rule_num = configs.fs_rule_num
        fsa_model_size = configs.fsa_model_size
        varpi_1 = configs.varpi_1
        varpi_2 = configs.varpi_2
        varpi_3 = configs.varpi_3
        delta_0 = configs.dropout
        tau_0 = configs.tau_0
        padding = patch_len - seq_len % patch_len
        att_channel_g = ceil((seq_len + padding) / patch_len)
        self.revin_layer = RevIN(input_channel)
        self.PatchData = PatchData(patch_len, padding)
        self.fuzzy_attention = FuzzyAttention(
            patch_len, fsa_rule_num, att_channel_g, fsa_model_size, varpi_1, varpi_2, delta_0, tau_0
        )
        self.Norm = nn.LayerNorm(att_channel_g)
        self.fuzzy_sys = FuzzyClass(
            att_channel_g * patch_len, pred_len, fs_rule_num, input_channel, delta_0, tau_0, varpi_3
        )

    def forward(self, x_enc):
        batch_size_now = x_enc.shape[0]
        x_enc = self.revin_layer(x_enc, "norm")
        x_enc = self.PatchData(x_enc)
        fs_out, loss_attention = self.fuzzy_attention(x_enc)
        x_enc = self.Norm(fs_out + x_enc)
        x_enc = Adjust_shape(x_enc, batch_size_now)
        x_out, loss_sys = self.fuzzy_sys(x_enc)
        x_out = self.revin_layer(x_out, "denorm")
        return x_out, loss_sys + loss_attention
