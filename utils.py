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

def compare_exps(exp_dirs, root_dir='/home/neil/AIR-ASVspoof/experiments'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    for folder in exp_dirs:
        out_fold = os.path.join(root_dir, folder)
        train_log_file = os.path.join(out_fold, "train_loss.log")
        dev_log_file = os.path.join(out_fold, "dev_loss.log")
        with open(train_log_file, "r") as train_log:
            x = np.array([[float(i) for i in line[0:-1].split('\t')] for line in train_log.readlines()[1:]])
            it_per_batch = int(x[:, 1].max()) + 1
            x = x[it_per_batch - 1::it_per_batch]
            ax1.plot(range(1, len(x) + 1), x[:, 2])
            #     ax1.set_xticks(np.where(x[:,1]==0)[0][::10])
            #     ax1.set_xticklabels(np.arange(np.sum(x[:,1]==0))*10)
            ax1.set_title("Training Loss")
            ax1.grid()
            ax1.legend(exp_dirs)
        with open(dev_log_file, "r") as dev_log:
            x = np.array([[float(i) for i in line[0:-1].split('\t')] for line in dev_log.readlines()[1:]])
            ax2.plot(range(1, len(x) + 1), x[:, 1])
            ax2.set_title("Validation Loss")
            ax2.legend(exp_dirs)

            ax3.plot(range(1, len(x) + 1), x[:, 2])
            ax3.set_title("Validation EER")
            ax3.minorticks_on()
            ax3.grid(b=True, which='major', linestyle='-')
            ax3.grid(b=True, which='minor', linestyle=':')
            ax3.legend(exp_dirs)
        with open(os.path.join(out_fold, "test_loss.log"), "r") as dev_eer:
            x = np.array([[float(i) for i in line[0:-1].split('\t')] for line in dev_eer.readlines()[1:]])
            ax4.plot(range(1, len(x) + 1), x[:, 2])
            ax4.set_title("Test EER")
            ax4.minorticks_on()
            ax4.grid(b=True, which='major', linestyle='-')
            ax4.grid(b=True, which='minor', linestyle=':')
            ax4.legend(exp_dirs)
    plt.show()

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
        X_tsne = TSNE(random_state=999, perplexity=40, early_exaggeration=100).fit_transform(X)
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


def test_checkpoint_model(feat_model_path, loss_model_path, part, add_loss, vis=False):
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
                            "LFCC", feat_len=750, pad_chop=False, padding="repeat")
    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
        for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            cqcc = cqcc.unsqueeze(1).float().to(device)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, cqcc_outputs = model(cqcc)

            score = cqcc_outputs[:, 0]

            ip1_loader.append(feats.detach().cpu())
            idx_loader.append((labels.detach().cpu()))
            tag_loader.append((tags.detach().cpu()))

            if add_loss in ["isolate", "iso_sq"]:
                score = torch.norm(feats - loss_model.center, p=2, dim=1)
            elif add_loss == "ang_iso":
                ang_isoloss, score = loss_model(feats, labels)

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'),
                                            "/data/neil/DS_10283_3336/")

    if vis:
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

def visualize_dev_and_eval(dev_feat, dev_labels, eval_feat, eval_labels, center, epoch, out_fold):
    # visualize which experiemnt which epoch
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
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
        X = np.concatenate((dev_feat, eval_feat), axis=0)
        X_tsne = TSNE(random_state=999, perplexity=40, early_exaggeration=100).fit_transform(X)
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

def get_features(feat_model_path, part):
    model = torch.load(feat_model_path)
    dataset = ASVspoof2019("LA", "/data/neil/DS_10283_3336/", "/dataNVME/neil/ASVspoof2019LAFeatures/",
                            "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part,
                            "LFCC", feat_len=750, pad_chop=False, padding="repeat")
    dataLoader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0,
                                collate_fn=dataset.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
    for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(dataLoader)):
        cqcc = cqcc.unsqueeze(1).float().to(device)
        tags = tags.to(device)
        labels = labels.to(device)
        feats, _ = model(cqcc)
        ip1_loader.append(feats.detach().cpu().numpy())
        idx_loader.append((labels.detach().cpu().numpy()))
        tag_loader.append((tags.detach().cpu().numpy()))
    features = np.concatenate(ip1_loader, 0)
    labels = np.concatenate(idx_loader, 0)
    gen_feats = features[labels==0]
    return features, labels, gen_feats

def predict_with_OCSVM(feat_model_path):
    from sklearn.svm import OneClassSVM
    _, _, train_gen_feats = get_features(feat_model_path, "train")
    clf = OneClassSVM(gamma='auto', nu=0.05).fit(train_gen_feats)
    dev_features, dev_labels, _ = get_features(feat_model_path, "dev")
    eval_features, eval_labels, _ = get_features(feat_model_path, "eval")
    dev_scores = clf.score_samples(dev_features)
    eval_scores = clf.score_samples(eval_features)
    dev_eer = em.compute_eer(dev_scores[dev_labels==0], dev_scores[dev_labels==1])
    eval_eer = em.compute_eer(eval_scores[eval_labels==0], eval_scores[eval_labels==1])

    clf2 = OneClassSVM(gamma='auto', nu=0.5).fit(train_gen_feats)
    dev_scores = clf2.score_samples(dev_features)
    eval_scores = clf2.score_samples(eval_features)
    dev_eer2 = em.compute_eer(dev_scores[dev_labels == 0], dev_scores[dev_labels == 1])
    eval_eer2 = em.compute_eer(eval_scores[eval_labels == 0], eval_scores[eval_labels == 1])

    clf3 = OneClassSVM(gamma='auto', nu=0.95).fit(train_gen_feats)
    dev_scores = clf3.score_samples(dev_features)
    eval_scores = clf3.score_samples(eval_features)
    dev_eer3 = em.compute_eer(dev_scores[dev_labels == 0], dev_scores[dev_labels == 1])
    eval_eer3 = em.compute_eer(eval_scores[eval_labels == 0], eval_scores[eval_labels == 1])

    return dev_eer[0], eval_eer[0], dev_eer2[0], eval_eer2[0], dev_eer3[0], eval_eer3[0]

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

    # feat_model_path = "/data/neil/antiRes/models1007/ce/anti-spoofing_cqcc_model.pt"
    # loss_model_path = "/data/neil/antiRes/models1007/ce/anti-spoofing_loss_model.pt"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # test_checkpoint_model(feat_model_path, loss_model_path, "eval", None)
    # print(eer)

    feat_model_path = "/data/neil/antiRes/models1015/ce/checkpoint/anti-spoofing_cqcc_model_%d.pt" % 80
    loss_model_path = "/data/neil/antiRes/models1015/ce/checkpoint/anti-spoofing_loss_model_%d.pt" % 80
    dev_eer, eval_eer, dev_eer2, eval_eer2, dev_eer3, eval_eer3 = predict_with_OCSVM(feat_model_path)
    print(dev_eer, eval_eer, dev_eer2, eval_eer2, dev_eer3, eval_eer3)
