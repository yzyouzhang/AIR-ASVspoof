import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
import model as model_
from resnet import ResNet
from dataset import ASVspoof2019
from torch.utils.data import DataLoader
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from loss import CenterLoss, LGMLoss_v0, LMCL_loss, IsolateLoss
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm, trange

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
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=16)

    parser.add_argument('-m', '--model', help='Model arch', required=True,
                        choices=['cnn', 'resnet', 'tdnn', 'lstm', 'rnn', 'cnn_lstm'], default='cnn')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=80, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=1024, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

    parser.add_argument('--add_loss', type=str, default=None, choices=['center', 'lgm', 'lgcl', 'isolate'], help="add other loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=0.0002, help="weight for other loss")

    parser.add_argument('--enable_tag', type=bool, default=False, help="use tags as multi-class label")
    parser.add_argument('--visualize', action='store_true', help="feature visualization")

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
    if args.model == 'cnn':
        # node_dict = {"CQCC": 128, "CQT": 256, "LFCC": 128, "MFCC": 256, "Melspec": 90}
        # cqcc_model = model_.CQCC_ConvNet(2, int(args.feat_len / 100 * node_dict[args.feat])).to(args.device)
        # node_dict = {"CQCC": 67840, "Melspec": 101760, "CQT": 156032, "STFT": 210304}
        # cqcc_model = model_.CQCC_ConvNet(2, node_dict[args.feat], subband_attention=False).to(args.device)
        node_dict = {"CQCC": 10, "Melspec": 15, "CQT": 156032, "STFT": 210304}
        cqcc_model = model_.CQCC_ConvNet(2, node_dict[args.feat], subband_attention=True).to(args.device)
    elif args.model == 'tdnn':
        node_dict = {"CQCC": 90, "CQT": 192, "LFCC": 60, "MFCC": 50}
        cqcc_model = model_.TDNN_classifier(node_dict[args.feat], 2).to(args.device)
    elif args.model == 'rnn':
        node_dict = {"CQCC": 90, "CQT": 192, "LFCC": 60, "MFCC": 50}
        cqcc_model = model_.RNN(node_dict[args.feat], 512, 2, 2).to(args.device)
    elif args.model == 'cnn_lstm':
        node_dict = {"CQCC": 90, "CQT": 192, "LFCC": 60, "MFCC": 50}
        cqcc_model = model_.CNN_LSTM(nclasses=2).to(args.device)
    elif args.model == 'resnet':
        node_dict = {"CQCC": 4, "Melspec": 6, "CQT": 8, "STFT": 11, "MFCC": 50}
        cqcc_model = ResNet(node_dict[args.feat], args.enc_dim, resnet_type='18', nclasses=2).to(args.device)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    cqcc_optimizer = torch.optim.Adam(cqcc_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'train',
                                args.feat, feat_len=args.feat_len)
    validation_set = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, 'dev',
                                  args.feat, feat_len=args.feat_len)
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    cqcc, _, _, _ = training_set[23]
    print("Feature shape", cqcc.shape)

    if args.add_loss == "center":
        centerLoss = CenterLoss(2, args.enc_dim).to(args.device)
        centerLoss.train()
        center_optimzer = torch.optim.SGD(centerLoss.parameters(), lr=0.5)

    if args.add_loss == "lgm":
        lgm_loss = LGMLoss_v0(2, args.enc_dim, 1.0).to(args.device)
        lgm_loss.train()
        lgm_optimzer = torch.optim.SGD(lgm_loss.parameters(), lr=0.1)

    if args.add_loss == "lgcl":
        lgcl_loss = LMCL_loss(2, args.enc_dim).to(args.device)
        lgcl_loss.train()
        lgcl_optimzer = torch.optim.SGD(lgcl_loss.parameters(), lr=0.01)

    if args.add_loss == "isolate":
        iso_loss = IsolateLoss(2, args.enc_dim).to(args.device)
        iso_loss.train()
        iso_optimzer = torch.optim.SGD(iso_loss.parameters(), lr=0.01)

    ip1_loader, idx_loader = [], []

    prev_loss = 1e8

    for epoch_num in tqdm(range(args.num_epochs)):
        cqcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        print('\nEpoch: %d ' % (epoch_num + 1))
        with trange(len(trainDataLoader)) as t:
            for i in t:
                cqcc, audio_fn, tags, labels = [d for d in next(iter(trainDataLoader))]
                if args.model == 'rnn':
                    cqcc = cqcc.transpose(1, 2).float().to(args.device)
                else:
                    cqcc = cqcc.unsqueeze(1).float().to(args.device)
                if not args.enable_tag:
                    labels = labels.to(args.device)
                    feats, cqcc_outputs = cqcc_model(cqcc)
                    cqcc_loss = criterion(cqcc_outputs, labels)

                else:
                    tags = tags.to(args.device)
                    feats, cqcc_outputs = cqcc_model(cqcc)
                    cqcc_loss = criterion(cqcc_outputs, tags)

                # Backward and optimize
                if args.add_loss == None:
                    cqcc_optimizer.zero_grad()
                    trainlossDict["feat_loss"].append(cqcc_loss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()

                if args.add_loss == "center":
                    centerloss = centerLoss(feats, labels)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    cqcc_loss += centerloss * args.weight_loss
                    cqcc_optimizer.zero_grad()
                    center_optimzer.zero_grad()
                    trainlossDict["center"].append(centerloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    # for param in centerLoss.parameters():
                    #     param.grad.data *= (1. / args.weight_loss)
                    center_optimzer.step()

                if args.add_loss == "isolate":
                    isoloss = iso_loss(feats, labels)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    cqcc_loss += isoloss * args.weight_loss
                    cqcc_optimizer.zero_grad()
                    iso_optimzer.zero_grad()
                    trainlossDict["iso"].append(isoloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    iso_optimzer.step()

                if args.add_loss == "lgm":
                    outputs, moutputs, likelihood = lgm_loss(feats, labels)
                    cqcc_loss = criterion(moutputs, labels)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    lgmloss = 0.5 * likelihood
                    # print(criterion(moutputs, labels).data, likelihood.data)
                    cqcc_optimizer.zero_grad()
                    lgm_optimzer.zero_grad()
                    cqcc_loss += lgmloss
                    trainlossDict["lgm"].append(lgmloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    lgm_optimzer.step()

                if args.add_loss == "lgcl":
                    outputs, moutputs = lgcl_loss(feats, labels)
                    cqcc_loss = criterion(moutputs, labels)
                    # print(criterion(moutputs, labels).data)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    cqcc_optimizer.zero_grad()
                    lgcl_optimzer.zero_grad()
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    lgcl_optimzer.step()

                ip1_loader.append(feats)
                idx_loader.append((labels))

                desc_str = ''
                for key in sorted(trainlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(trainlossDict[key])) + ', '
                t.set_description(desc_str)

        if args.visualize:
            feat = torch.cat(ip1_loader, 0)
            labels = torch.cat(idx_loader, 0)
            visualize(args, feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch_num+1)

        # Val the model
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        cqcc_model.eval()

        with torch.no_grad():
            cqcc_correct = 0

            total = 0

            with trange(len(valDataLoader)) as v:
                for i in v:
                    cqcc, audio_fn, tags, labels = [d for d in next(iter(valDataLoader))]
                    if args.model == 'rnn':
                        cqcc = cqcc.transpose(1, 2).float().to(args.device)
                    else:
                        cqcc = cqcc.unsqueeze(1).float().to(args.device)
                    labels = labels.to(args.device)
                    feats, cqcc_outputs = cqcc_model(cqcc)
                    cqcc_loss = criterion(cqcc_outputs, labels)
                    devlossDict["feat_loss"].append(cqcc_loss.item())
                    if args.add_loss in [None, "center", "lgcl", "isolate"]:
                        _, cqcc_predicted = torch.max(cqcc_outputs.data, 1)
                        total += labels.size(0)
                        cqcc_correct += (cqcc_predicted == labels).sum().item()
                    elif args.add_loss == "lgm":
                        outputs, moutputs, likelihood = lgm_loss(feats, labels)
                        _, cqcc_predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        cqcc_correct += (cqcc_predicted == labels).sum().item()

                    # if (k+1) % 10 == 0:
                    #     print('Epoch [{}/{}], Step [{}/{}], cqcc_accuracy {:.4f} %'.format(
                    #         epoch_num + 1, args.num_epochs, k + 1, len(valDataLoader), (100 * cqcc_correct / total)))

                    desc_str = ''
                    for key in sorted(devlossDict.keys()):
                        desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                    v.set_description(desc_str)
            valLoss = np.nanmean(devlossDict['feat_loss'])

            if valLoss < prev_loss:
                # Save the model checkpoint
                torch.save(cqcc_model, os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt'))
                if args.add_loss == "center":
                    loss_model = centerLoss
                    torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                elif args.add_loss == "isolate":
                    loss_model = iso_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                elif args.add_loss == "lgm":
                    loss_model = lgm_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                elif args.add_loss == "lgcl":
                    loss_model = lgcl_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                else:
                    loss_model = None
                prev_loss = valLoss
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt == 6:
                break
            # if early_stop_cnt == 1:
            #     torch.save(cqcc_model, os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt')

            print('Dev Accuracy of the model on the val features: {} % '.format(100 * cqcc_correct / total))

    return cqcc_model, loss_model


def test(args, model, loss_model, part='eval'):
    test_set = ASVspoof2019(args.path_to_database, args.path_to_features, args.path_to_protocol, part,
                            args.feat, feat_len=args.feat_len)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size // 2, shuffle=False, num_workers=args.num_workers)
    cqcc_correct = 0
    total = 0
    with open(os.path.join(args.out_fold, 'cm_score.txt'), 'w') as cm_score_file:
        for i, (cqcc, audio_fn, tags, labels) in enumerate(testDataLoader):
            if args.model == 'rnn':
                cqcc = cqcc.transpose(1, 2).float().to(args.device)
            else:
                cqcc = cqcc.unsqueeze(1).float().to(args.device)
            feats, cqcc_outputs = model(cqcc)
            labels = labels.to(args.device)

            if args.add_loss in [None, "center", "isolate"]:
                _, cqcc_predicted = torch.max(cqcc_outputs.data, 1)
                total += labels.size(0)
                cqcc_correct += (cqcc_predicted == labels).sum().item()
            elif args.add_loss == "lgm":
                cqcc_outputs, moutputs, likelihood = loss_model(feats, labels)
                _, cqcc_predicted = torch.max(cqcc_outputs.data, 1)
                total += labels.size(0)
                cqcc_correct += (cqcc_predicted == labels).sum().item()
            elif args.add_loss == "lgcl":
                cqcc_outputs, moutputs = loss_model(feats, labels)
                _, cqcc_predicted = torch.max(cqcc_outputs.data, 1)
                total += labels.size(0)
                cqcc_correct += (cqcc_predicted == labels).sum().item()

            if (i + 1) % 20 == 0:
                print('Step [{}/{}] '.format(i + 1, len(testDataLoader)))
                print('Test Accuracy of the model on the eval features: {} %'.format(100 * cqcc_correct / total))
            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          cqcc_outputs.data[j][0].cpu().numpy()))
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
