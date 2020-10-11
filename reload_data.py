import pickle
from librosa.util import find_files
import scipy.io as sio
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np

# local
# path_to_features = '/home/yzh298/anti-spoofing/ASVspoof2019Features/'

access_type = "LA"
# # on air station gpu
path_to_mat = '/dataNVME/neil/LALFBmat/'
path_to_audio = '/data/neil/DS_10283_3336/'+access_type+'/ASVspoof2019_'+access_type+'_'
path_to_features = '/dataNVME/neil/ASVspoof2019'+access_type+'Features/'
# on son gpu
# path_to_mat = '/home/neil/ASVspoof2019'+access_type+'Features/'
# path_to_audio = '/home/neil/DS_10283_3336/'+access_type+'/ASVspoof2019_'+access_type+'_'
# path_to_features = '/home/neil/ASVspoof2019'+access_type+'Features/'

def reload_data(path_to_features, part):
    matfiles = find_files(path_to_mat + part + '/', ext='mat')
    for i in range(len(matfiles)):
        if matfiles[i][len(path_to_mat)+len(part)+1:].startswith('CQCC'):
            key = matfiles[i][len(path_to_mat) + len(part) + 6:-4]
            cqcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
            with open(path_to_features + part +'/'+ key + 'CQCC.pkl', 'wb') as handle1:
                pickle.dump(cqcc, handle1, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            key = matfiles[i][len(path_to_mat) + len(part) + 6:-4]
            lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
            with open(path_to_features + part +'/'+ key + 'LFCC.pkl', 'wb') as handle2:
                pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)

def reload_lfb(path_to_features, part):
    matfiles = find_files(path_to_mat + part + '/', ext='mat')
    for i in range(len(matfiles)):
        key = matfiles[i][len(path_to_mat) + len(part) + 5:-4]
        cqcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
        with open(path_to_features + part +'/'+ key + 'LFB.pkl', 'wb') as handle1:
            pickle.dump(cqcc, handle1, protocol=pickle.HIGHEST_PROTOCOL)

def reload_mfcc(path_to_audio, path_to_features, part):
    audiofiles = find_files(path_to_audio + part + '/flac/', ext='flac')
    for i in range(len(audiofiles)):
        audio, sr = librosa.load(audiofiles[i], sr=16000, mono=True)
        key = audiofiles[i][len(path_to_audio + part + '/flac/'):-5]
        mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=50, n_fft=512, hop_length=256)
        with open(path_to_features + part + '/' + key + 'MFCC.pkl', 'wb') as handle0:
            pickle.dump(mfcc, handle0, protocol=pickle.HIGHEST_PROTOCOL)

def reload_cqt(path_to_audio, path_to_features, part):
    audio_files = find_files(path_to_audio + part + '/flac/', ext='flac')
    for i in range(len(audio_files)):
        audio, sr = librosa.load(audio_files[i], sr=16000, mono=True)
        key = audio_files[i][len(path_to_audio + part + '/flac/'):-5]
        cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=128, fmin=16000/(2^10), n_bins=96*2, bins_per_octave=96))
        # print(cqt_dict[key].shape)
        # plt.pcolormesh(cqt_dict[key], cmap=plt.cm.viridis)
        # plt.savefig(os.path.join("./models/try/", str(i)+".png"))
        # plt.close()
        with open(path_to_features + part + '/' + key + 'CQT.pkl', 'wb') as handle3:
            pickle.dump(cqt, handle3, protocol=pickle.HIGHEST_PROTOCOL)

def reload_stft(path_to_audio, path_to_features, part):
    audio_files = find_files(path_to_audio + part + '/flac/', ext='flac')
    for i in range(len(audio_files)):
        audio, sr = librosa.load(audio_files[i], sr=16000, mono=True)
        key = audio_files[i][len(path_to_audio + part + '/flac/'):-5]
        stft = np.abs(librosa.stft(audio, n_fft=512))   #,128 Mel
        with open(path_to_features + part + '/' + key + 'STFT.pkl', 'wb') as handle4:
            pickle.dump(stft, handle4, protocol=pickle.HIGHEST_PROTOCOL)


def reload_melspec(path_to_audio, path_to_features, part):
    audio_files = find_files(path_to_audio + part + '/flac/', ext='flac')
    for i in range(len(audio_files)):
        audio, sr = librosa.load(audio_files[i], sr=16000, mono=True)
        key = audio_files[i][len(path_to_audio + part + '/flac/'):-5]
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512, hop_length=128)
        with open(path_to_features + part +'/'+ key +'Melspec.pkl', 'wb') as handle5:
            pickle.dump(melspec, handle5, protocol=pickle.HIGHEST_PROTOCOL)

def reload_wavform(path_to_audio, path_to_features, part):
    audio_files = find_files(path_to_audio + part + '/flac/', ext='flac')
    for i in range(len(audio_files)):
        audio, sr = librosa.load(audio_files[i], sr=16000, mono=True)
        key = audio_files[i][len(path_to_audio + part + '/flac/'):-5]
        raw_wav = audio
        with open(path_to_features + part + '/' + key + 'RawWav.pkl', 'wb') as handle6:
            pickle.dump(raw_wav, handle6, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == "__main__":
    # reload_data(path_to_features, 'train')
    # reload_data(path_to_features, 'dev')
    # reload_data(path_to_features, 'eval')
    for part in ["train", "dev", "eval"]:
        reload_lfb(path_to_features, part)
        # reload_mfcc(path_to_audio, path_to_features, part)
        # reload_cqt(path_to_audio, path_to_features, part)
        # reload_stft(path_to_audio, path_to_features, part)
        # reload_melspec(path_to_audio, path_to_features, part)
        # reload_wavform(path_to_audio, path_to_features, part)


