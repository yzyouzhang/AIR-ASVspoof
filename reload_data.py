import pickle
from librosa.util import find_files
import scipy.io as sio

access_type = "LA"
# on air station gpu
path_to_mat = 'D:/Users/Suchit/Desktop/Acad/EED 305 Digital Signal Processing/DSP Project/DS_10283_3336/anti-spoofing/ASVspoof2019/LA/Features/'
#generally the same as the path_to_features.
path_to_audio = 'D:/Users/Suchit/Desktop/Acad/EED 305 Digital Signal Processing/DSP Project/DS_10283_3336/' + access_type + '/ASVspoof2019_' + access_type + '_'
#path to all folders inside LA folder of ASVspoof2019 dataset
path_to_features = 'D:/Users/Suchit/Desktop/Acad/EED 305 Digital Signal Processing/DSP Project/DS_10283_3336/anti-spoofing/ASVspoof2019/' + access_type + '/Features/'
#path to the .m files obtained as output after executing process_LA_data.m

def reload_data(path_to_features, part):
    matfiles = find_files(path_to_mat + part + '/', ext='mat')
    for i in range(len(matfiles)):
        if matfiles[i][len(path_to_mat) + len(part) + 1:].startswith('LFCC'):
            key = matfiles[i][len(path_to_mat) + len(part) + 6:-4]
            print("Currently processing ", key)
            lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
            with open(path_to_features + part + '/' + key + 'LFCC.pkl', 'wb') as handle2:
                pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    reload_data(path_to_features, 'train')
    reload_data(path_to_features, 'dev')
    reload_data(path_to_features, 'eval')
    print("If the code ran for very short time, then there is some error with paths or something, and the .pkl files were not created as expected in 'with open' line (line 21)")
    print("If the code ran and you saw many 'Currently processing' lines, it is successful.")