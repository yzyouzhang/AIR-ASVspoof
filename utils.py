import json
import os
import re
import pandas as pd

def read_args_json(model_path):
    with open(os.path.join(model_path, "args.json"), 'r') as json_file:
        content = json_file.readlines()
        x = "".join(content[:-5]).replace('\n', ',')[:-1]
        args = json.loads(x, strict=False)
        y = "".join(content[-5:])
        a, b, c, d, e, f = [float(res) for res in re.findall("0\.\d+", y)]
    return args, (a, b, c, d, e, f)

def create_new_split(df_train, df_dev, split_dict):
    for split_key in split_dict:
        os.system("mkdir " + os.path.join(os.getcwd(), 'traindev_split', split_key))
        with open(os.path.join(os.getcwd(), 'traindev_split', split_key,
                               'ASVspoof2019.LA.cm.train.trl.txt'), 'w') as new_split_train_file, \
                open(os.path.join(os.getcwd(), 'traindev_split', split_key,
                                  'ASVspoof2019.LA.cm.dev.trl.txt'), 'w') as new_split_dev_file:
            for i in range(len(df_train)):
                speaker, filename, _, attack, label = df_train.iloc[i]
                if attack == "-":
                    new_split_train_file.write('%s %s - %s bonafide\n' % (speaker, filename, attack))
                elif attack in split_dict[split_key]:
                    new_split_train_file.write('%s %s - %s spoof\n' % (speaker, filename, attack))
                else:
                    new_split_dev_file.write('%s %s - %s spoof\n' % (speaker, filename, attack))
            for i in range(len(df_dev)):
                speaker, filename, _, attack, label = df_dev.iloc[i]
                if attack == "-":
                    new_split_dev_file.write('%s %s - %s bonafide\n' % (speaker, filename, attack))
                elif attack in split_dict[split_key]:
                    new_split_train_file.write('%s %s - %s spoof\n' % (speaker, filename, attack))
                else:
                    new_split_dev_file.write('%s %s - %s spoof\n' % (speaker, filename, attack))


if __name__ == "__main__":
    args, (a, b, c, d, e, f) = read_args_json("models/cqt_cnn1")
    split_dict = {'125_346': ["A01", "A02", "A05"], '135_246': ["A01", "A03", "A05"], '145_236': ["A01", "A04", "A05"],
                  '235_146': ["A02", "A03", "A05"], '245_136': ["A02", "A04", "A05"], '345_126': ["A03", "A04", "A05"],
                  '126_345': ["A01", "A02", "A06"], '136_245': ["A01", "A03", "A06"], '146_235': ["A01", "A04", "A06"],
                  '236_145': ["A02", "A03", "A06"], '246_135': ["A02", "A04", "A06"], '346_125': ["A03", "A04", "A06"]}
    df_train = pd.read_csv("/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm." + "train" + ".trl.txt",
                     names=["speaker", "filename", "-", "attack", "label"], sep=" ")
    df_dev = pd.read_csv("/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm." + "dev" + ".trl.txt",
                     names=["speaker", "filename", "-", "attack", "label"], sep=" ")
    create_new_split(df_train, df_dev, split_dict)

