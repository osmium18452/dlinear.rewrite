import torch
import torch.nn as nn
import torch.nn.functional as F
from FEDLayers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_pos_temp, DataEmbedding_wo_temp
from FEDLayers.AutoCorrelation import AutoCorrelationLayer
from FEDLayers.FourierCorrelation import FourierBlock, FourierCrossAttention
from FEDLayers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from FEDLayers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, \
    series_decomp_multi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, pred_len,
                 output_attention=False, d_model=512, moving_avg=24, embed='timeF', freq='h', dropout=.05, modes=64,
                 mode_select='random', n_heads=8, d_ff=2048, activation='gelu', d_layers=1, e_layers=2):
        '''
        :param enc_in: encoder input channel
        :param dec_in: decoder input channel
        :param c_out: output channel
        :param seq_len: input length
        :param pred_len: output length
        :param output_attention:
        :param d_model:
        :param moving_avg:
        :param embed:
        :param freq:
        :param dropout:
        :param modes:
        :param mode_select:
        '''
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        encoder_self_att = FourierBlock(in_channels=d_model,
                                        out_channels=d_model,
                                        seq_len=self.seq_len,
                                        modes=modes,
                                        mode_select_method=mode_select)
        decoder_self_att = FourierBlock(in_channels=d_model,
                                        out_channels=d_model,
                                        seq_len=self.seq_len // 2 + self.pred_len,
                                        modes=modes,
                                        mode_select_method=mode_select)
        decoder_cross_att = FourierCrossAttention(in_channels=d_model,
                                                  out_channels=d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=modes,
                                                  mode_select_method=mode_select)
        # Encoder
        enc_modes = int(min(modes, seq_len // 2))
        dec_modes = int(min(modes, (seq_len // 2 + pred_len) // 2))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        d_model, n_heads),

                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # print('x enc shape',x_enc.shape,x_mark_enc.shape)
        # exit()
        # decomp init
        label_len = x_enc.shape[1] // 2
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        x_mark_dec=torch.cat([x_mark_enc[:, -label_len:, :], x_mark_dec], dim=1)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

