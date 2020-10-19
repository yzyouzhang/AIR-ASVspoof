import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        #self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):

        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0)==1:
            attentions = F.softmax(torch.tanh(weights),dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5*torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)

            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

            representations = torch.cat((avg_repr,std_repr),1)

            return representations


class CQCC_ConvNet(nn.Module):
    def __init__(self, num_classes=2, num_nodes=512, enc_dim=2, subband_attention=False):
        super(CQCC_ConvNet, self).__init__()

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=7, stride=3, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(kernel_size=3, stride=3))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(kernel_size=3, stride=3))
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 2), stride=(2, 3), bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=3, stride=3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=3, stride=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(num_nodes, 3), padding=(0, 1), dilation=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(num_nodes, 256)
        self.fc2 = nn.Linear(256, enc_dim)
        self.fc3 = nn.Linear(enc_dim, num_classes)
        # self.dropout = nn.Dropout(0.2)

        self.subband_attention = subband_attention
        if self.subband_attention:
            self.attention = SelfAttention(128)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        # print(h.shape)
        if self.subband_attention:
            # print(h.shape)
            h = self.layer5(h)
            h = h.squeeze(2)
            # print(h.shape)
            h = self.attention(h.permute(0, 2, 1).contiguous())
            # print(h.shape)
            out = h
        else:
            h = h.reshape(h.size(0), -1)
            out = self.fc1(h)
        # h = self.dropout(h)
        out1 = self.fc2(out)
        out = self.fc3(out1)

        return out1, out

class DOC_ConvNet(nn.Module):
    def __init__(self, num_classes=2, num_nodes=512, subband_attention=True, feat_dim=2):
        super(DOC_ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 2), stride=(2, 3), bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )

        self.fc1 = nn.Linear(num_nodes, 128)
        self.fc2 = nn.Linear(128, feat_dim)
        self.fc3 = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

        self.subband_attention = subband_attention
        if self.subband_attention:
            self.attention = None

    def forward_once(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = h.reshape(h.size(0), -1)
        # h = self.dropout(h)
        out = self.fc1(h)
        out1 = self.fc2(out)
        out = self.fc3(out1)

        return out1, out

    def forward(self, ref, tar):
        feats_ref, out_ref = self.forward_once(ref)
        feats_tar, out_tar = self.forward_once(tar)

        return feats_ref, out_ref, feats_tar, out_tar


class TDNN_layer(nn.Module):
    def __init__(
            self,
            input_dim=23,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True,
            dropout_p=0.0
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN_layer, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)
        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        if self.dropout_p:
            x = self.drop(x)
        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        return x


class TDNN_classifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(TDNN_classifier, self).__init__()
        self.tdnn1 = TDNN_layer(input_dim=input_dim, output_dim=512, context_size=5, dilation=1)
        self.tdnn2 = TDNN_layer(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.tdnn3 = TDNN_layer(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.tdnn4 = TDNN_layer(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.tdnn5 = TDNN_layer(input_dim=512, output_dim=150, context_size=1, dilation=1)
        self.fc1 = nn.Linear(300, 2)
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        h = self.tdnn1(x)
        h = self.tdnn2(h)
        h = self.tdnn3(h)
        h = self.tdnn4(h)
        out = self.tdnn5(h)
        # print(out.shape)
        mean = torch.mean(out, 1)
        std = torch.var(out, 1)
        feats = torch.cat((mean, std), 1)
        feats = self.fc1(feats)
        out = self.fc2(feats)

        return feats, out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 2)
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        # Forward propagate LSTM
        out1, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out1 = self.fc1(out1[:, -1, :])
        out = self.fc2(out1)
        return out1, out


class Attention(nn.Module):
    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha


class CNN_LSTM(nn.Module):
    def __init__(self, n_z=256, nclasses=-1):
        super(CNN_LSTM, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5,5), padding=(1,2), dilation=(1,2), stride=(2,2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), padding=(1,2), dilation=(1,2), stride=(2,2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), padding=(1,2), dilation=(1,1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(5,5), padding=(1,2), dilation=(1,1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU() )

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5,5), padding=(1,2), dilation=(1,2), stride=(2,1), bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=3, stride=3)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(5,5), padding=(1,2), dilation=(1,2), stride=(2,1), bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=3, stride=3)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5,5), padding=(1,2), dilation=(1,1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1), dilation=(1,1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_fin = nn.Conv2d(64, 128, kernel_size=(10, 3), stride=(1,1), padding=(0,1), bias=False)
        self.bn_fin = nn.BatchNorm2d(128)

        self.lstm = nn.LSTM(128, 128, 2, bidirectional=True, batch_first=False)

        self.fc_feat = nn.Linear(128*2, 2)

        self.fc_mu = nn.Linear(2, nclasses) if nclasses>=2 else nn.Linear(2, 1)

        self.initialize_params()

    def forward(self, x):

        y = self.layer1(x)
        # print(y.shape)
        y = self.layer2(y)
        # print(y.shape)
        y = self.layer3(y)
        # print(y.shape)
        y = self.layer4(y)
        # print(y.shape)

        # x = self.features(x)
        # print(x.shape)
        x = self.conv_fin(y)
        # print(x.shape)
        feats = F.relu(self.bn_fin(x)).squeeze(2)
        # print(feats.shape)
        feats = feats.permute(2,0,1)
        batch_size = feats.size(1)
        # seq_size = feats.size(0)

        h0 = torch.zeros(2*2, batch_size, 128)
        c0 = torch.zeros(2*2, batch_size, 128)

        if x.is_cuda:
            h0 = h0.to(x.device)
            c0 = c0.to(x.device)

        out_seq, h_c = self.lstm(feats, (h0, c0))

        out_end = out_seq.mean(0)

        feat = self.fc_feat(out_end)

        mu = self.fc_mu(feat)

        return feat, mu


    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight)
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


if __name__ == "__main__":
    cqcc = torch.randn((32,1,128,650))
    # cnn_lstm = CNN_LSTM(nclasses=2)
    cnn = CQCC_ConvNet(num_classes=2, num_nodes=11, subband_attention=True)
    _, output = cnn(cqcc)
    print(output.shape)
    # print(output)

# def shape_list(x):
#     """Return list of dims, statically where possible."""
#     static = x.get_shape().as_list()
#     shape = x.shape
#     ret = []
#     for i, static_dim in enumerate(static):
#         dim = static_dim or shape[i]
#     ret.append(dim)
#     return ret
#
# def split_heads_2_d(inputs, Nh):
#     """Split channels into multiple heads."""
#     B, H, W, d = shape
#     list(inputs)
#     ret_shape = [B, H, W, Nh, d // Nh]
#     split = tf.reshape(inputs, ret_shape)
#     return tf.transpose(split, [0, 3, 1, 2, 4])
#
# def combine_heads_2_d(inputs):
#     """Combine heads (inverse of split heads 2d)."""
#     transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
#     Nh, channels = shape
#     list(transposed)[−2:]
#     ret_shape = shape
#     list(transposed)[:−2] + [Nh ∗ channels]
#     return tf.reshape(transposed, ret_shape)
#
# def rel_to_abs(x):
#     """Converts tensor from relative to aboslute indexing."""
#     # [B, Nh, L, 2L−1]
#     B, Nh, L, = shape_list(x)
#     # Pad to shift from relative to absolute indexing.
#     col_pad = tf.zeros((B, Nh, L, 1))
#     x = tf.concat([x, col pad], axis=3)
#     flat_x = tf.reshape(x, [B, Nh, L ∗ 2 ∗ L])
#     flat_pad = tf.zeros((B, Nh, L−1))
#     flat_x_padded = tf.concat([flat x, flat pad], axis=2)
#     # Reshape and slice out the padded elements.
#     final_x = tf.reshape(flat_x_padded, [B, Nh, L + 1, 2∗L−1])
#     final_x = final_x[:, :, :L, L−1:]
#     return final_x
