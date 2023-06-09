import torch
import torch.nn as nn
from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .embed import DataEmbedding, CustomEmbedding
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct


class Encoder(nn.Module):
    """ An encoder model with self attention mechanism. """

    def __init__(self, d_model, window_size, truncate, decoder, input_size, device, enc_in,
                 inner_size=3, d_inner_hid=512, n_head=8, d_k=128, d_v=128, dropout=.05, use_tvm=False, n_layer=4,
                 embed_type='CustomEmbedding', CSCM='Bottleneck_Construct', d_bottleneck=128):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.truncate = truncate
        if decoder == 'attention':
            self.mask, self.all_size = get_mask(input_size, window_size, inner_size, device)
        else:
            self.mask, self.all_size = get_mask(input_size + 1, window_size, inner_size, device)
        self.decoder_type = decoder
        if decoder == 'FC':
            self.indexes = refer_points(self.all_size, window_size, device)

        if use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            padding = 1 if decoder == 'FC' else 0
            q_k_mask = get_q_k(input_size + padding, inner_size, window_size[0], device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout,
                             normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in
                range(n_layer)
            ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout,
                             normalize_before=False) for i in range(n_layer)
            ])

        if embed_type == 'CustomEmbedding':
            self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
            # self.enc_embedding = CustomEmbedding(enc_in, d_model, covariate_size, seq_num, dropout)
        else:
            self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        # print('opt.cscm',CSCM)
        # print(d_model,window_size,d_bottleneck)
        # exit()
        self.conv_layers = eval(CSCM)(d_model, window_size, d_bottleneck)

    def forward(self, x_enc, x_mark_enc):

        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        # print(mask.shape)
        # exit()
        # print('......',seq_enc.shape)
        seq_enc = self.conv_layers(seq_enc)
        # print('......',seq_enc.shape)
        # exit()

        for i in range(len(self.layers)):
            # print(seq_enc.shape)
            # exit()
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, enc_in, out_len, input_size, device,
                 d_model=256, decoder='FC', window_size=None, truncate=False):
        super().__init__()
        # print(vars(opt))
        # exit()

        if window_size is None:
            window_size = [4, 4, 4]
        self.predict_step = out_len
        self.d_model = d_model
        self.input_size = input_size
        self.decoder_type = decoder

        self.encoder = Encoder(d_model, window_size, truncate, decoder, input_size, device, enc_in)
        if decoder == 'attention':
            mask = get_subsequent_mask(input_size, window_size, out_len, truncate)
            self.decoder = Decoder(mask, enc_in)
            self.predictor = Predictor(d_model, enc_in)
        elif decoder == 'FC':
            self.predictor = Predictor(4 * d_model, out_len * enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain=False):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

            if pretrain:
                dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
                pred = self.predictor(dec_enc)
            else:
                pred = self.predictor(dec_enc)
        elif self.decoder_type == 'FC':
            # batch_x = torch.cat([batch_x, predict_token], dim=1)
            # batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)

            predict_token = torch.zeros(x_enc.size(0), 1, x_enc.size(-1), device=x_enc.device)
            x_enc=torch.cat([x_enc,predict_token],dim=1)
            x_mark_enc=torch.cat([x_mark_enc,x_mark_dec[:,0:1,:]],dim=1)

            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)

        return pred
