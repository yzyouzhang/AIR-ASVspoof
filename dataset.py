import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pickle
import os
from torch.utils.data.dataloader import default_collate
import pandas as pd

torch.set_default_tensor_type(torch.FloatTensor)

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_database, path_to_features, path_to_protocol, part='train', feature='CQCC',
                 genuine_only=False, feat_len=650, pad_chop=True, padding='zero'):
        self.access_type = access_type
        self.ptd = path_to_database
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        if self.part == "eval":
            protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
                                    '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        # # would not work if change data split but this csv is only for feat_len
        # self.csv = pd.read_csv(self.ptf + "Set_csv.csv")

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            if genuine_only:
                assert self.part in ["train", "dev"]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[:num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        try:
            with open(self.ptf + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            def the_other(train_or_dev):
                assert train_or_dev in ["train", "dev"]
                res = "dev" if train_or_dev == "train" else "train"
                return res
            with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)

        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        # assert self.csv.at[idx, "feat_len"] == feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            feat_mat = feat_mat[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        return feat_mat, filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)
        # if self.pad_chop:
        #     feat_mat_lst, audio_fn_lst, tag_lst, label_lst = [], [], [], []
        #     for sample in samples:
        #         feat_mat, audio_fn, tag, label = sample
        #         if feat_mat.shape[1] % self.feat_len < self.feat_len * 0.5:
        #             num_of_items = feat_mat.shape[1] // self.feat_len
        #             for i in range(num_of_items):
        #                 feat_mat_lst.append(feat_mat[:, self.feat_len * i: self.feat_len * (i + 1)])
        #                 audio_fn_lst.append(audio_fn)
        #                 tag_lst.append(tag)
        #                 label_lst.append(label)
        #         else:
        #             num_of_items = feat_mat.shape[1] // self.feat_len + 1
        #             for i in range(num_of_items - 1):
        #                 feat_mat_lst.append(feat_mat[:, self.feat_len * i: self.feat_len * (i + 1)])
        #                 audio_fn_lst.append(audio_fn)
        #                 tag_lst.append(tag)
        #                 label_lst.append(label)
        #             feat_mat_lst.append(padding(feat_mat[:, self.feat_len * (num_of_items - 1):], self.feat_len))
        #             audio_fn_lst.append(audio_fn)
        #             tag_lst.append(tag)
        #             label_lst.append(label)
        #
        #     return default_collate(feat_mat_lst), default_collate(audio_fn_lst), \
        #            default_collate(tag_lst), default_collate(label_lst)
        # else:
        #     feat_mat = [sample[0].transpose(0, 1) for sample in samples]
        #     from torch.nn.utils.rnn import pad_sequence
        #     feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
        #     audio_fn = [sample[1] for sample in samples]
        #     tag = [sample[2] for sample in samples]
        #     label = [sample[3] for sample in samples]
        #
        #     return feat_mat, default_collate(audio_fn), default_collate(tag), default_collate(label)


def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


if __name__ == "__main__":
    path_to_database = '/data/neil/DS_10283_3336/'  # if run on GPU
    path_to_features = '/dataNVME/neil/ASVspoof2019Features/'  # if run on GPU
    path_to_protocol = '/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/'
    training_set = ASVspoof2019(path_to_database, path_to_features, path_to_protocol, genuine_only=False, pad_chop=False, feature='Melspec', feat_len=320)
    feat_mat, audio_fn, tag, label = training_set[2999]
    print(len(training_set))
    print(audio_fn)
    print(feat_mat.shape)
    # print(cqcc.shape)
    # print(lfcc.shape)
    print(tag)
    print(label)
    # samples = [training_set[26], training_set[27], training_set[28], training_set[29]]
    # out = training_set.collate_fn(samples)

    # training_set = ASVspoof2019(path_to_database, path_to_features)
    # cqcc, audio_fn, tag, label = training_set[2580]
    # print(len(training_set))
    # print(audio_fn)
    # # print(mfcc.shape)
    # print(cqcc.shape)
    # # print(lfcc.shape)
    # print(tag)
    # print(label)

    trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0, collate_fn=training_set.collate_fn)
    feat_mat_batch, audio_fn, tags, labels = [d for d in next(iter(trainDataLoader))]
    print(feat_mat_batch.shape)
    # print(feat_mat_batch)
