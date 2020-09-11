import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pickle
import os

torch.set_default_tensor_type(torch.FloatTensor)


class ASVspoof2019(Dataset):
    def __init__(self, path_to_database, path_to_features, path_to_protocol, part='train', feature='CQCC', genuine_only=False, feat_len=650, pad_chop=True):
        self.ptd = path_to_database
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.path_to_audio = os.path.join(self.ptd, 'LA/ASVspoof2019_LA_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.pad_chop = pad_chop
        self.path_to_protocol = path_to_protocol
        # protocol = self.ptd+'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.'+ self.part +'.trl.txt'
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.LA.cm.'+ self.part + '.trl.txt')
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        with open(self.ptf + feature + 'FeatureMat.pkl', 'rb') as cqcc_handle:
            self.cqcc_mat = pickle.load(cqcc_handle)

        ## add this if statement since we may change the data split
        if self.part == "train":
            with open(os.path.join(path_to_features, "dev") + feature + 'FeatureMat.pkl', 'rb') as cqcc_other_handle:
                self.other_cqcc_mat = pickle.load(cqcc_other_handle)
        elif self.part == "dev":
            with open(os.path.join(path_to_features, "train") + feature + 'FeatureMat.pkl', 'rb') as cqcc_other_handle:
                self.other_cqcc_mat = pickle.load(cqcc_other_handle)
        else:
            protocol = self.ptd + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.' + self.part + '.trl.txt'
            self.other_cqcc_mat = None
        self.returned_lst = preprocessing(protocol, self.cqcc_mat, self.other_cqcc_mat, self.feat_len, self.genuine_only, self.part, self.pad_chop)

    def __len__(self):
        return len(self.returned_lst)

    def __getitem__(self, idx):
        cqcc, audio_fn, tag, label = self.returned_lst[idx]
        return cqcc, audio_fn, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        from torch.utils.data.dataloader import default_collate
        if self.pad_chop:
            return default_collate(samples)
        else:
            cqcc = [sample[0].transpose(0, 1) for sample in samples]
            from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence
            cqcc, lengths = pad_packed_sequence(pack_sequence(cqcc, enforce_sorted=False), True)
            audio_fn = [sample[1] for sample in samples]
            tag = [sample[2] for sample in samples]
            label = [sample[3] for sample in samples]

            # return pack_padded_sequence(cqcc, lengths, batch_first=True, enforce_sorted=False), \
            #        default_collate(audio_fn), default_collate(tag), default_collate(label)
            return cqcc, default_collate(audio_fn), default_collate(tag), default_collate(label)


def preprocessing(protocol, cqcc_mat, other_cqcc_mat, feat_len, genuine_only, part, pad_chop):
    total_len = 0
    lst = []
    with open(protocol, 'r') as f:
        audio_info = [info.strip().split() for info in f.readlines()]
        if genuine_only:
            num_bonafide = {"train": 2580, "dev": 2548}
            all_info = enumerate(audio_info[:num_bonafide[part]])
        else:
            all_info = enumerate(audio_info)
        for idx, info in all_info:
            audio_fn = info[1]
            try:
                cqcc = torch.from_numpy(cqcc_mat[audio_fn])
            except:
                cqcc = torch.from_numpy(other_cqcc_mat[audio_fn])
            if pad_chop:
                if cqcc.shape[1] % feat_len < feat_len * 0.5:
                    num_of_items = cqcc.shape[1] // feat_len
                    total_len += num_of_items
                    for i in range(num_of_items):
                        lst.append((cqcc[:, feat_len * i: feat_len * (i + 1)], audio_fn, info[3], info[4]))  # cqcc, filename, tag, label
                else:
                    num_of_items = cqcc.shape[1] // feat_len + 1
                    total_len += num_of_items
                    for i in range(num_of_items - 1):
                        lst.append((cqcc[:, feat_len * i: feat_len*(i+1)], audio_fn, info[3], info[4])) # cqcc, filename, tag, label
                    lst.append((padding(cqcc[:, feat_len * (num_of_items - 1):], feat_len), audio_fn, info[3], info[4]))
            else:
                lst.append((cqcc, audio_fn, info[3], info[4]))
    return lst


def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    # print(spec.dtype)
    # print(torch.zeros(width, padd_len).dtype)
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)


if __name__ == "__main__":
    path_to_database = '/data/neil/DS_10283_3336/'  # if run on GPU
    path_to_features = '/data/neil/ASVspoof2019Features/'  # if run on GPU
    training_set = ASVspoof2019(path_to_database, path_to_features, genuine_only=True, pad_chop=False)
    cqcc, audio_fn, tag, label = training_set[26]
    print(len(training_set))
    print(audio_fn)
    # print(mfcc.shape)
    print(cqcc.shape)
    # print(lfcc.shape)
    print(tag)
    print(label)
    samples = [training_set[26], training_set[27], training_set[28], training_set[29]]
    out = training_set.collate_fn(samples)

    # training_set = ASVspoof2019(path_to_database, path_to_features)
    # cqcc, audio_fn, tag, label = training_set[2580]
    # print(len(training_set))
    # print(audio_fn)
    # # print(mfcc.shape)
    # print(cqcc.shape)
    # # print(lfcc.shape)
    # print(tag)
    # print(label)

