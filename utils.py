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

from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from collections import defaultdict
from tqdm import tqdm
import eval_metrics as em

# lr 10 0.1
# r 0.2 0.9
# Adam SGD

def visualize(feat, tags, labels, center, epoch, trainOrDev, out_fold):
    # visualize which experiemnt which epoch
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
    # plt.ion()
    # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    c = ['#ff0000', '#003366', '#ffff00']
    c_tag = ['#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900',
             '#009900', '#009999', '#00ff00', '#990000', '#999900', '#ff0000',
             '#003366', '#ffff00', '#f0000f', '#0f00f0', '#00ffff', '#0000ff', '#ff00ff']
    # plt.clf()
    num_centers, enc_dim = center.shape
    if enc_dim > 2:
        X = np.concatenate((center, feat), axis=0)
        X_tsne = TSNE(random_state=999).fit_transform(X)
        center = X_tsne[:num_centers]
        feat = X_tsne[num_centers:]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        ex_ratio = pca.explained_variance_ratio_
        center_pca = X_pca[:num_centers]
        feat_pca = X_pca[num_centers:]
    else:
        center_pca = center
        feat_pca = feat
        ex_ratio = [0.5, 0.5]
    # t-SNE visualization
    ax1.plot(feat[labels == 0, 0], feat[labels == 0, 1], '.', c=c[0], markersize=1)
    ax1.plot(feat[labels == 1, 0], feat[labels == 1, 1], '.', c=c[1], markersize=1)
    ax1.plot(center[:, 0], center[:, 1], 'x', c=c[2], markersize=5)
    plt.setp((ax2, ax3), xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.plot(feat[labels == 0, 0], feat[labels == 0, 1], '.', c=c[0], markersize=2)
    for i in range(1, 20):
        ax3.plot(feat[tags == i, 0], feat[tags == i, 1], '.', c=c_tag[i-1], markersize=2)
    ax3.legend(['A01', 'A02', 'A03', 'A04', 'A05', 'A06',
                'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'])
    ax1.legend(['genuine', 'spoofing', 'center'])
    # PCA visualization
    ax4.plot(feat_pca[labels == 0, 0], feat_pca[labels == 0, 1], '.', c=c[0], markersize=1)
    ax4.plot(feat_pca[labels == 1, 0], feat_pca[labels == 1, 1], '.', c=c[1], markersize=1)
    ax4.plot(center_pca[:, 0], center_pca[:, 1], 'x', c=c[2], markersize=5)
    plt.setp((ax5, ax6), xlim=ax4.get_xlim(), ylim=ax4.get_ylim())
    ax5.plot(feat_pca[labels == 0, 0], feat_pca[labels == 0, 1], '.', c=c[0], markersize=2)
    for i in range(1, 20):
        ax6.plot(feat_pca[tags == i, 0], feat_pca[tags == i, 1], '.', c=c_tag[i - 1], markersize=2)
    ax6.legend(['A01', 'A02', 'A03', 'A04', 'A05', 'A06',
                'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'])
    ax4.legend(['genuine', 'spoofing', 'center'])
    fig.suptitle("Feature Visualization of Epoch %d, %s, %.5f, %.5f" % (epoch, trainOrDev, ex_ratio[0], ex_ratio[0]))
    plt.savefig(os.path.join(out_fold, trainOrDev + '_vis_feat_epoch=%d.jpg' % epoch))
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

def test_checkpoint_model(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
        epoch_num = int(basename.split("_")[-1])
    else:
        dir_path = dirname(feat_model_path)
        epoch_num = 0
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set = ASVspoof2019("LA", "/data/neil/DS_10283_3336/", "/dataNVME/neil/ASVspoof2019LAFeatures/",
                            "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part,
                            "Melspec", feat_len=750, pad_chop=False)
    testDataLoader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ip1_loader, tag_loader, idx_loader = [], [], []
    model.eval()
    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            cqcc = cqcc.unsqueeze(1).float().to(device)
            feats, cqcc_outputs = model(cqcc)
            ip1_loader.append(feats.detach().cpu())
            tags = tags.to(device)
            tag_loader.append(tags.detach().cpu())
            labels = labels.to(device)
            idx_loader.append(labels.detach().cpu())
            for j in range(labels.size(0)):
                if add_loss == "isolate":
                    score = torch.norm(feats[j].unsqueeze(0) - loss_model.center, p=2, dim=1).data.item()
                elif add_loss == "ang_iso":
                    score = F.normalize(feats[j].unsqueeze(0), p=2, dim=1) @ F.normalize(loss_model.center, p=2,
                                                                                         dim=1).T
                    score = score.data.item()
                elif add_loss == "multi_isolate":
                    genuine_dist = torch.norm((feats[j].unsqueeze(0).repeat(args.num_centers, 1) - loss_model.centers),
                                              p=2, dim=1)
                    score, indices = torch.min(genuine_dist, dim=-1)
                    score = score.item()
                elif add_loss == "multicenter_isolate":
                    score = 1e8
                    for k in range(loss_model.centers.shape[0]):
                        dist = torch.norm(feats[j] - loss_model.centers[k].unsqueeze(0), p=2, dim=1)
                        if dist.item() < score:
                            score = dist.item()
                else:
                    score = F.softmax(cqcc_outputs.data[j])[0].cpu().numpy()
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score))
    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'),
                                            "/data/neil/DS_10283_3336/")
    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    tags = torch.cat(tag_loader, 0)
    if add_loss == "isolate":
        centers = loss_model.center
    elif add_loss == "multi_isolate":
        centers = loss_model.centers
    elif add_loss == "ang_iso":
        centers = loss_model.center
    else:
        centers = torch.mean(feat[labels == 0], dim=0, keepdim=True)
    torch.save(feat, os.path.join(dir_path, 'feat_%d.pt' % epoch_num))
    torch.save(tags, os.path.join(dir_path, 'tags_%d.pt' % epoch_num))
    visualize(feat.data.cpu().numpy(), tags.data.cpu().numpy(), labels.data.cpu().numpy(),
              centers.data.cpu().numpy(), epoch_num, part, dir_path)

    return eer_cm, min_tDCF

def test_fluctuate_eer(feat_model_path, loss_model_path, part, add_loss, visualize=False):
    dirname=os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
        epoch_num = int(basename.split("_")[-1])
    else:
        dir_path = dirname(feat_model_path)
        epoch_num = 0
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set = ASVspoof2019("LA","/data/neil/DS_10283_3336/", "/dataNVME/neil/ASVspoof2019LAFeatures/",
                            "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part,
                            "Melspec", feat_len=750, pad_chop=False)
    testDataLoader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
    model.eval()
    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            cqcc = cqcc.unsqueeze(1).float().to(device)
            feats, cqcc_outputs = model(cqcc)
            ip1_loader.append(feats.detach().cpu())
            tags = tags.to(device)
            tag_loader.append(tags.detach().cpu())
            labels = labels.to(device)
            idx_loader.append(labels.detach().cpu())
            # score = torch.norm(feats - loss_model.center, p=2, dim=1)
            score = cqcc_outputs[:, 0]
            score_loader.append(score.detach().cpu())
        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        eer, threshold = em.compute_eer(scores[labels==0], scores[labels==1])
    return eer

if __name__ == "__main__":
    # args, (a, b, c, d, e, f) = read_args_json("models/cqt_cnn1")
    #
    # split_dict = {'125_346': ["A01", "A02", "A05"], '135_246': ["A01", "A03", "A05"], '145_236': ["A01", "A04", "A05"],
    #               '235_146': ["A02", "A03", "A05"], '245_136': ["A02", "A04", "A05"], '345_126': ["A03", "A04", "A05"],
    #               '126_345': ["A01", "A02", "A06"], '136_245': ["A01", "A03", "A06"], '146_235': ["A01", "A04", "A06"],
    #               '236_145': ["A02", "A03", "A06"], '246_135': ["A02", "A04", "A06"], '346_125': ["A03", "A04", "A06"]}
    # df_train = pd.read_csv("/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm." + "train" + ".trl.txt",
    #                  names=["speaker", "filename", "-", "attack", "label"], sep=" ")
    # df_dev = pd.read_csv("/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm." + "dev" + ".trl.txt",
    #                  names=["speaker", "filename", "-", "attack", "label"], sep=" ")
    # create_new_split(df_train, df_dev, split_dict)

    # feat_model_path = "/home/neil/AIR-ASVspoof/models0922/ang_iso/checkpoint/anti-spoofing_cqcc_model_19.pt"
    # loss_model_path = "/home/neil/AIR-ASVspoof/models0922/ang_iso/checkpoint/anti-spoofing_loss_model_19.pt"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # test_checkpoint_model(feat_model_path, loss_model_path, "eval", "ang_iso")

    # b = []
    # for i in range(50, 120, 2):
    #     feat_model_path = "/data/neil/antiRes/models1007/iso_loss4/checkpoint/anti-spoofing_cqcc_model_%d.pt" % i
    #     loss_model_path = "/data/neil/antiRes/models1007/iso_loss4/checkpoint/anti-spoofing_loss_model_%d.pt" % i
    #     eer = test_fluctuate_eer(feat_model_path, loss_model_path, "eval", "isolate")
    #     print(eer)
    #     b.append(eer)

    feat_model_path = "/data/neil/antiRes/models1007/ce/anti-spoofing_cqcc_model.pt"
    loss_model_path = "/data/neil/antiRes/models1007/ce/anti-spoofing_loss_model.pt"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_checkpoint_model(feat_model_path, loss_model_path, "eval", None)
    # print(eer)
