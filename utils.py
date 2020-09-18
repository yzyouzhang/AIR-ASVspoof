import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import numpy as np

def visualize(args, feat, labels, epoch):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex='col', sharey='col')
    # plt.ion()
    # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    c = ['#ff0000', '#003366']
    # plt.clf()
    ax1.plot(feat[labels == 0, 0], feat[labels == 0, 1], '.', c=c[0], markersize=1)
    ax1.plot(feat[labels == 1, 0], feat[labels == 1, 1], '.', c=c[1], markersize=1)
    plt.setp((ax2, ax3), xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.plot(feat[labels == 0, 0], feat[labels == 0, 1], '.', c=c[0], markersize=2)
    ax3.plot(feat[labels == 1, 0], feat[labels == 1, 1], '.', c=c[1], markersize=2)
    fig.legend(['genuine', 'spoofing'], loc='upper right')
    #   plt.xlim(xmin=-5,xmax=5)
    #   plt.ylim(ymin=-5,ymax=5)
    fig.suptitle("Feature Visualization of Epoch %d" % epoch)
    plt.savefig(os.path.join(args.out_fold, 'vis_loss_epoch=%d.jpg' % epoch))
    plt.show()
    fig.clf()
    plt.close(fig)

def seek_centers_kmeans(args, num_centers, genuine_dataloader, model):
    model.eval()
    genuine_feats = []
    with torch.no_grad():
        for i, (feat, _, _, _) in enumerate(genuine_dataloader):
            feat = feat.unsqueeze(1).float().to(args.device)
            new_feats, _ = model(feat)
            genuine_feats.append(new_feats)
        kmeans = KMeans(n_clusters=num_centers, init='k-means++', random_state=0).fit(torch.cat(genuine_feats, 0).data.cpu().numpy())
        centers = torch.from_numpy(kmeans.cluster_centers_).to(args.device)
        model.train()
    return centers

def read_args_json(model_path):
    with open(os.path.join(model_path, "args.json"), 'r') as json_file:
        content = json_file.readlines()
        x = "".join(content[:-5]).replace('\n', ',')[:-1]
        args = json.loads(x, strict=False)
        y = "".join(content[-5:])
        a, b, c, d, e, f = [float(res) for res in re.findall("0\.\d+", y)]
    return args, (a, b, c, d, e, f)

def plot_loss(args):
    log_file = os.path.join(args.out_fold, "loss.log")
    with open(log_file, "r") as log:
        x = np.array([[float(i) for i in line[:-1].split('\t')] for line in log.readlines()])
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
        ax1.plot(x[:, 1])
        ax1.set_title("Discriminator Loss")
        ax2.plot(abs(x[:, 2]))
        ax2.set_title("Generator Loss")
    return fig

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

