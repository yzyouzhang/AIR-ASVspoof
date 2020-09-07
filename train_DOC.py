import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import DOC_ConvNet
from resnet import DOC_ResNet
from dataset import ASVspoof2019
from torch.utils.data import DataLoader
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from loss import CenterLoss, LGMLoss_v0, LMCL_loss
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm, trange
from sklearn.svm import OneClassSVM

torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=999)

    # Data folder prepare
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path", default='/data/neil/DS_10283_3336/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/dataNVME/neil/ASVspoof2019Features/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use",
                        choices=["CQCC", "LFCC", "MFCC", "STFT", "Melspec", "CQT"], default='CQCC')
    parser.add_argument("--feat_len", type=int, help="features length", default=650)
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=2)

    parser.add_argument('-m', '--model', help='Model arch', required=True,
                        choices=['cnn', 'resnet'], default='cnn')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=8, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=1024, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=3, help="number of workers")

    parser.add_argument('--add_loss', type=str, default=None, help="add other loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=0.0002, help="weight for other loss")

    parser.add_argument('--enable_tag', type=bool, default=False, help="use tags as multi-class label")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Path for output data
    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)

    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
    else:
        shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
        os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

    # Path for input data
    assert os.path.exists(args.path_to_database)
    assert os.path.exists(args.path_to_features)

    # Save training arguments
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if int(args.gpu) == 5:
        args.device = torch.device("cpu")

    return args

def visualize(args, feat, labels, epoch):
    plt.ion()
    # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    c = ['#ff0000', '#00ff00']
    plt.clf()
    for i in range(2):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    #   plt.xlim(xmin=-5,xmax=5)
    #   plt.ylim(ymin=-5,ymax=5)
    plt.text(-4.8, 3.6, "epoch=%d" % epoch)
    plt.savefig(os.path.join(args.out_fold, 'vis_loss_epoch=%d.jpg' % epoch))
    plt.show()
    plt.close()

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    # node_dict = {"CQCC": 128, "CQT": 256, "LFCC": 128, "MFCC": 256}
    # cqcc_model = DOC_ConvNet(2, int(args.feat_len / 100 * node_dict[args.feat]), feat_dim=args.enc_dim).to(args.device)
    if args.model == 'cnn':
        node_dict = {"CQCC": 67840, "Melspec": 101760, "CQT": 156032, "STFT": 210304}
        cqcc_model = DOC_ConvNet(2, node_dict[args.feat]).to(args.device)
    elif args.model == 'resnet':
        node_dict = {"CQCC": 4, "Melspec": 6, "CQT": 8, "STFT": 11, "MFCC": 50}
        cqcc_model = DOC_ResNet(node_dict[args.feat], resnet_type='18', nclasses=2).to(args.device)

    # Loss and optimizer
    descriptive_loss = nn.CrossEntropyLoss()
    cqcc_optimizer = torch.optim.Adam(cqcc_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'train',
                                args.feat, feat_len=args.feat_len)
    validation_set = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'dev',
                                  args.feat, feat_len=args.feat_len)
    training_genuine = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'train', args.feat,
                                    feat_len=args.feat_len, genuine_only=True)
    validation_genuine = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'dev', args.feat,
                                      feat_len=args.feat_len, genuine_only=True)
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    trainGenDataLoader = DataLoader(training_genuine, batch_size=args.batch_size // 8, shuffle=True, num_workers=args.num_workers)
    valGenDataLoader = DataLoader(validation_genuine, batch_size=args.batch_size // 8, shuffle=True, num_workers=args.num_workers)

    compactness_loss = CenterLoss(2, args.enc_dim).to(args.device)
    compactness_loss.train()
    compactness_optimzer = torch.optim.SGD(compactness_loss.parameters(), lr=0.5)

    ip1_loader, idx_loader = [], []
    prev_loss = 1e8

    for epoch_num in tqdm(range(args.num_epochs)):
        cqcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        print('\nEpoch: %d ' % (epoch_num + 1))
        # with trange(2) as t:
        with trange(len(trainDataLoader)) as t:
            for i in t:
                cqcc, audio_fn, tags, labels = next(iter(trainDataLoader))
                cqcc_gen, audio_fn_gen, tags_gen, labels_gen = next(iter(trainGenDataLoader))
                cqcc = cqcc.unsqueeze(1).float().to(args.device)
                cqcc_gen = cqcc_gen.unsqueeze(1).float().to(args.device)

                labels = labels.to(args.device)
                labels_gen = labels_gen.to(args.device)
                feats, cqcc_outputs, feats_gen, outputs_gen = cqcc_model(cqcc, cqcc_gen)
                l_d = descriptive_loss(cqcc_outputs, labels)
                l_c = compactness_loss(feats_gen, labels_gen)
                trainlossDict["l_d"].append(l_d.item())
                trainlossDict["l_c"].append(l_c.item())
                cqcc_loss = l_d + l_c * args.weight_loss

                # Backward and optimize
                cqcc_optimizer.zero_grad()
                compactness_optimzer.zero_grad()
                cqcc_loss.backward()
                cqcc_optimizer.step()
                compactness_optimzer.step()

                ip1_loader.append(feats)
                idx_loader.append((labels))

                desc_str = ''
                for key in sorted(trainlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(trainlossDict[key])) + ', '
                t.set_description(desc_str)

        feat = torch.cat(ip1_loader, 0)
        labels = torch.cat(idx_loader, 0)

        visualize(args, feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch_num+1)

        # Val the model
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        cqcc_model.eval()

        with torch.no_grad():
            cqcc_correct = 0
            total = 0
            # with trange(2) as v:
            with trange(len(valDataLoader)) as v:
                for i in v:
                    cqcc, audio_fn, tags, labels = [d for d in next(iter(valDataLoader))]
                    cqcc_gen, audio_fn_gen, tags_gen, labels_gen = next(iter(valGenDataLoader))
                    cqcc = cqcc.unsqueeze(1).float().to(args.device)
                    cqcc_gen = cqcc_gen.unsqueeze(1).float().to(args.device)
                    labels = labels.to(args.device)
                    labels_gen = labels_gen.to(args.device)
                    feats, cqcc_outputs, feats_gen, outputs_gen = cqcc_model(cqcc, cqcc_gen)
                    l_d = descriptive_loss(cqcc_outputs, labels)
                    l_c = compactness_loss(feats_gen, labels_gen)
                    devlossDict["l_d"].append(l_d.item())
                    devlossDict["l_c"].append(l_c.item())
                    # cqcc_loss = l_d + l_c * args.weight_loss
                    _, cqcc_predicted = torch.max(cqcc_outputs.data, 1)
                    total += labels.size(0)
                    cqcc_correct += (cqcc_predicted == labels).sum().item()

                    desc_str = ''
                    for key in sorted(devlossDict.keys()):
                        desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                    v.set_description(desc_str)

            valLoss = np.nanmean(devlossDict['l_d']) + np.nanmean(devlossDict['l_c']) * args.weight_loss / 5

            if valLoss < prev_loss:
                # Save the model checkpoint
                torch.save(cqcc_model, os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt'))
                loss_model = compactness_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                prev_loss = valLoss
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt == 6:
                break

            print('Dev Accuracy of the model on the val features: {} % '.format(100 * cqcc_correct / total))

        # # Save the model checkpoint
        # loss_model = compactness_loss
        # torch.save(cqcc_model, os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt'))
        # torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
    return cqcc_model, loss_model

def fit_OCSVM(args, model):
    # training_set = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'train', args.feat,
    #                             feat_len=args.feat_len)
    training_genuine = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'train', args.feat,
                                    feat_len=args.feat_len, genuine_only=True)
    # validation_genuine = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'dev', args.feat,
    #                                   feat_len=args.feat_len, genuine_only=True)
    # trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    trainGenDataLoader = DataLoader(training_genuine, batch_size=len(training_genuine), shuffle=True,
                                    num_workers=args.num_workers)
    clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    with torch.no_grad():
        with trange(len(trainGenDataLoader)) as t:
            for i in t:
                # cqcc, _, _, _ = next(iter(trainDataLoader))
                cqcc_gen, _, _, _ = next(iter(trainGenDataLoader))

                # cqcc = cqcc.unsqueeze(1).float().to(args.device)
                cqcc_gen = cqcc_gen.unsqueeze(1).float().to(args.device)

                _, _, feats_gen, _ = model(cqcc_gen, cqcc_gen)
                clf.fit(feats_gen.data.cpu().numpy())
    return clf


def test(args, model, loss_model, part='eval'):
    test_set = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, part, args.feat, feat_len=args.feat_len)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    cqcc_correct = 0
    total = 0
    # clf = fit_OCSVM(args, model)

    with open(os.path.join(args.out_fold, 'cm_score.txt'), 'w') as cm_score_file:
        for i, (cqcc, audio_fn, tags, labels) in enumerate(testDataLoader):
            cqcc = cqcc.unsqueeze(1).float().to(args.device)
            feats, cqcc_outputs, _, _ = model(cqcc, cqcc)
            labels = labels.to(args.device)

            _, cqcc_predicted = torch.max(cqcc_outputs.data, 1)
            total += labels.size(0)
            cqcc_correct += (cqcc_predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print('Step [{}/{}] '.format(i + 1, len(testDataLoader)))
                print('Test Accuracy of the model on the eval features: {} %'.format(100 * cqcc_correct / total))
            # score_list = clf.score_samples(feats.data.cpu().numpy())
            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          cqcc_outputs.data[j][0].cpu().numpy()))
                # cm_score_file.write(
                #     '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                #                           "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                #                           score_list[j]))
    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(args.out_fold, 'cm_score.txt'), args)

    return eer_cm, min_tDCF


if __name__ == "__main__":
    args = initParams()
    _, loss_model = train(args)
    model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt'))
    # loss_model = None
    TReer_cm, TRmin_tDCF = test(args, model, loss_model, "train")
    VAeer_cm, VAmin_tDCF = test(args, model, loss_model, "dev")
    TEeer_cm, TEmin_tDCF = test(args, model, loss_model)
    with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
        res_file.write('\nTrain EER: %8.5f min-tDCF: %8.5f\n' % (TReer_cm, TRmin_tDCF))
        res_file.write('\nVal EER: %8.5f min-tDCF: %8.5f\n' % (VAeer_cm, VAmin_tDCF))
        res_file.write('\nTest EER: %8.5f min-tDCF: %8.5f\n' % (TEeer_cm, TEmin_tDCF))
