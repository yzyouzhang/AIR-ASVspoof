import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize(args, feat, tags, labels, center, epoch, trainOrDev):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
    # plt.ion()
    # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    c = ['#ff0000', '#003366', '#ffff00']
    c_tag = ['#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900']
    # plt.clf()
    if args.enc_dim > 2:
        X = np.concatenate((center, feat), axis=0)
        X_tsne = TSNE(random_state=args.seed).fit_transform(X)
        center = X_tsne[0][np.newaxis, :]
        feat = X_tsne[1:]
        X_pca = PCA(n_components=2).fit_transform(X)
        center_pca = X_pca[0][np.newaxis, :]
        feat_pca = X_pca[1:]
    else:
        center_pca = center
        feat_pca = feat
    # t-SNE visualization
    ax1.plot(feat[labels == 0, 0], feat[labels == 0, 1], '.', c=c[0], markersize=1)
    ax1.plot(feat[labels == 1, 0], feat[labels == 1, 1], '.', c=c[1], markersize=1)
    ax1.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    plt.setp((ax2, ax3), xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.plot(feat[labels == 0, 0], feat[labels == 0, 1], '.', c=c[0], markersize=2)
    for i in range(1, 7):
        ax3.plot(feat[tags == i, 0], feat[tags == i, 1], '.', c=c_tag[i-1], markersize=2)
    ax3.legend(['A01', 'A02', 'A03', 'A04', 'A05', 'A06'])
    ax1.legend(['genuine', 'spoofing', 'center'])
    # PCA visualization
    ax4.plot(feat_pca[labels == 0, 0], feat_pca[labels == 0, 1], '.', c=c[0], markersize=1)
    ax4.plot(feat_pca[labels == 1, 0], feat_pca[labels == 1, 1], '.', c=c[1], markersize=1)
    ax4.plot(center_pca[:, 0], center_pca[:, 1], 'x', c=c[2], markersize=5)
    plt.setp((ax5, ax6), xlim=ax4.get_xlim(), ylim=ax4.get_ylim())
    ax5.plot(feat_pca[labels == 0, 0], feat_pca[labels == 0, 1], '.', c=c[0], markersize=2)
    for i in range(1, 7):
        ax6.plot(feat_pca[tags == i, 0], feat_pca[tags == i, 1], '.', c=c_tag[i - 1], markersize=2)
    ax6.legend(['A01', 'A02', 'A03', 'A04', 'A05', 'A06'])
    ax4.legend(['genuine', 'spoofing', 'center'])
    fig.suptitle("Feature Visualization of Epoch %d, %s" % (epoch, trainOrDev))
    plt.savefig(os.path.join(args.out_fold, trainOrDev + '_vis_feat_epoch=%d.jpg' % epoch))
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
    train_log_file = os.path.join(args.out_fold, "train_loss.log")
    dev_log_file = os.path.join(args.out_fold, "dev_loss.log")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    with open(train_log_file, "r") as train_log:
        x = np.array([[float(i) for i in line[0:-1].split('\t')] for line in train_log.readlines()[1:]])
        ax1.plot(x[:, 2])
        ax1.set_xticks(np.where(x[:, 1] == 0)[0][::10])
        ax1.set_xticklabels(np.arange(np.sum(x[:, 1] == 0)) * 10)
        ax1.set_title("Training Loss")
    with open(dev_log_file, "r") as dev_log:
        x = np.array([[float(i) for i in line[0:-1].split('\t')] for line in dev_log.readlines()[1:]])
        ax2.plot(x[:, 1])
        ax2.set_title("Validation Loss")
    plt.savefig(os.path.join(args.out_fold, 'loss_curve.jpg'))
    plt.show()
    fig.clf()
    plt.close(fig)
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

