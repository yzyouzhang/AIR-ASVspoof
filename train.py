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
from loss import *
from collections import defaultdict
from tqdm import tqdm, trange
from utils import *
import eval_metrics as em

torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=999)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path", default='/data/neil/DS_10283_3336/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/dataNVME/neil/ASVspoof2019LAFeatures/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use", required=True,
                        choices=["CQCC", "LFCC", "MFCC", "STFT", "Melspec", "CQT", "LFB"], default='Melspec')
    parser.add_argument("--feat_len", type=int, help="features length", default=650)
    parser.add_argument('--pad_chop', type=bool, default=False, help="whether pad_chop in the dataset")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=16)

    parser.add_argument('-m', '--model', help='Model arch', required=True,
                        choices=['cnn', 'resnet', 'tdnn', 'lstm', 'rnn', 'cnn_lstm'], default='resnet')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.8, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

    parser.add_argument('--add_loss', type=str, default=None,
                        choices=[None, 'center', 'lgm', 'lgcl', 'isolate', 'ang_iso', 'multi_isolate', 'multicenter_isolate'], help="add other loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real', type=float, default=0.5, help="r_real for isolate loss")
    parser.add_argument('--r_fake', type=float, default=30, help="r_fake for isolate loss")
    parser.add_argument('--num_centers', type=int, default=3, help="num of centers for multi isolate loss")

    parser.add_argument('--enable_tag', type=bool, default=False, help="use tags as multi-class label")
    parser.add_argument('--visualize', action='store_true', help="feature visualization")
    parser.add_argument('--test_only', action='store_true', help="test the trained model in case the test crash sometimes or another test method")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    parser.add_argument('--pre_train', action='store_true', help="whether to pretrain the model")
    parser.add_argument('--prtrn_mthd', type=str, default='cross_entropy',
                            choices=['cross_entropy', 'single_center'], help="pretrain method, in other words, pretrain with what loss")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.test_only or args.continue_training:
        pass
    else:
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

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if int(args.gpu) == 5:
        args.device = torch.device("cpu")

    return args

def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'resnet':
        node_dict = {"CQCC": 4, "LFCC": 3, "Melspec": 6, "CQT": 8, "STFT": 11, "MFCC": 87}
        cqcc_model = ResNet(node_dict[args.feat], args.enc_dim, resnet_type='18', nclasses=2).to(args.device)

    if args.continue_training:
        cqcc_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt')).to(args.device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    cqcc_optimizer = torch.optim.Adam(cqcc_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ASVspoof2019(args.access_type, args.path_to_database, args.path_to_features, args.path_to_protocol, 'train',
                                args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop)
    genuine_trainset = ASVspoof2019(args.access_type, args.path_to_database, args.path_to_features, args.path_to_protocol, 'train',
                                args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, genuine_only=True)
    validation_set = ASVspoof2019(args.access_type, args.path_to_database, args.path_to_features, args.path_to_protocol, 'dev',
                                  args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop)
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=training_set.collate_fn)
    genuine_trainDataLoader = DataLoader(genuine_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=genuine_trainset.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               collate_fn=validation_set.collate_fn)

    feat, _, _, _ = training_set[23]
    print("Feature shape", feat.shape)

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
        iso_loss = IsolateLoss(2, args.enc_dim, r_real=args.r_real, r_fake=args.r_fake).to(args.device)
        if args.continue_training:
            iso_loss = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt')).to(args.device)
        iso_loss.train()
        iso_optimzer = torch.optim.SGD(iso_loss.parameters(), lr=0.01)

    if args.add_loss == "ang_iso":
        ang_iso = AngularIsoLoss(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=10).to(args.device)
        ang_iso.train()
        ang_iso_optimzer = torch.optim.SGD(ang_iso.parameters(), lr=0.01)

    if args.add_loss == "multi_isolate":
        multi_iso_loss = MultiIsolateCenterLoss(args.enc_dim, args.num_centers, r_real=args.r_real, r_fake=args.r_fake).to(args.device)
        if args.continue_training:
            multi_iso_loss = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt')).to(args.device)
        multi_iso_loss.train()
        multi_iso_optimzer = torch.optim.SGD(multi_iso_loss.parameters(), lr=0.01)

    if args.add_loss == "multicenter_isolate":
        centers = torch.randn((3, args.enc_dim)) * 10
        centers = centers.to(args.device)
        if args.pre_train:
            cqcc_model, cqcc_optimizer = pre_train(args, trainDataLoader, cqcc_model, cqcc_optimizer)
            centers = seek_centers_kmeans(args, 3, genuine_trainDataLoader, cqcc_model)

    early_stop_cnt = 0
    prev_loss = 1e8

    for epoch_num in tqdm(range(args.num_epochs)):
        genuine_feats, ip1_loader, tag_loader, idx_loader = [], [], [], []
        cqcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        adjust_learning_rate(args, cqcc_optimizer, epoch_num)
        if args.add_loss == "isolate":
            adjust_learning_rate(args, iso_optimzer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        # with trange(2) as t:
        with trange(len(trainDataLoader)) as t:
            for i in t:
                cqcc, audio_fn, tags, labels = [d for d in next(iter(trainDataLoader))]
                cqcc = cqcc.unsqueeze(1).float().to(args.device)
                tags = tags.to(args.device)
                labels = labels.to(args.device)
                feats, cqcc_outputs = cqcc_model(cqcc)
                cqcc_loss = criterion(cqcc_outputs, labels)

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
                    trainlossDict[args.add_loss].append(centerloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    # for param in centerLoss.parameters():
                    #     param.grad.data *= (1. / args.weight_loss)
                    center_optimzer.step()

                if args.add_loss == "isolate":
                    isoloss = iso_loss(feats, labels)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    cqcc_loss = isoloss * args.weight_loss
                    cqcc_optimizer.zero_grad()
                    iso_optimzer.zero_grad()
                    trainlossDict[args.add_loss].append(isoloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    iso_optimzer.step()

                if args.add_loss == "ang_iso":
                    ang_isoloss = ang_iso(feats, labels)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    cqcc_loss = ang_isoloss * args.weight_loss
                    cqcc_optimizer.zero_grad()
                    ang_iso_optimzer.zero_grad()
                    trainlossDict[args.add_loss].append(ang_isoloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    ang_iso_optimzer.step()

                if args.add_loss == "multi_isolate":
                    multi_isoloss = multi_iso_loss(feats, labels)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    cqcc_loss = multi_isoloss * args.weight_loss
                    cqcc_optimizer.zero_grad()
                    multi_iso_optimzer.zero_grad()
                    trainlossDict[args.add_loss].append(multi_isoloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    multi_iso_optimzer.step()

                if args.add_loss == "multicenter_isolate":
                    multicenter_iso_loss = MultiCenterIsolateLoss(centers, 2, args.enc_dim, r_real=args.r_real, r_fake=args.r_fake).to(
                        args.device)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    multiisoloss = multicenter_iso_loss(feats, labels)
                    cqcc_loss = multiisoloss * args.weight_loss
                    cqcc_optimizer.zero_grad()
                    trainlossDict[args.add_loss].append(multiisoloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()

                if args.add_loss == "lgm":
                    outputs, moutputs, likelihood = lgm_loss(feats, labels)
                    cqcc_loss = criterion(moutputs, labels)
                    trainlossDict["feat"].append(cqcc_loss.item())
                    lgmloss = 0.5 * likelihood
                    # print(criterion(moutputs, labels).data, likelihood.data)
                    cqcc_optimizer.zero_grad()
                    lgm_optimzer.zero_grad()
                    cqcc_loss += lgmloss
                    trainlossDict[args.add_loss].append(lgmloss.item())
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    lgm_optimzer.step()

                if args.add_loss == "lgcl":
                    outputs, moutputs = lgcl_loss(feats, labels)
                    cqcc_loss = criterion(moutputs, labels)
                    # print(criterion(moutputs, labels).data)
                    trainlossDict[args.add_loss].append(cqcc_loss.item())
                    cqcc_optimizer.zero_grad()
                    lgcl_optimzer.zero_grad()
                    cqcc_loss.backward()
                    cqcc_optimizer.step()
                    lgcl_optimzer.step()

                # genuine_feats.append(feats[labels==0])
                ip1_loader.append(feats)
                idx_loader.append((labels))
                tag_loader.append((tags))

                desc_str = ''
                for key in sorted(trainlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(trainlossDict[key])) + ', '
                t.set_description(desc_str)

                with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" + str(i) + "\t" + str(np.nanmean(trainlossDict[args.add_loss])) + "\n")

        if args.add_loss == "multicenter_isolate":
            centers = seek_centers_kmeans(args, 3, genuine_trainDataLoader, cqcc_model)

        if args.visualize and ((epoch_num+1) % 3 == 1):
            feat = torch.cat(ip1_loader, 0)
            labels = torch.cat(idx_loader, 0)
            tags = torch.cat(tag_loader, 0)
            if args.add_loss == "isolate":
                centers = iso_loss.center
            elif args.add_loss == "multi_isolate":
                centers = multi_iso_loss.centers
            elif args.add_loss == "ang_iso":
                centers = ang_iso.center
            else:
                centers = torch.mean(feat[labels == 0], dim=0, keepdim=True)
            visualize(args, feat.data.cpu().numpy(), tags.data.cpu().numpy(), labels.data.cpu().numpy(), centers.data.cpu().numpy(),
                      epoch_num + 1, "Train")

        # Val the model
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        cqcc_model.eval()

        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
            # with trange(2) as v:
            with trange(len(valDataLoader)) as v:
                for i in v:
                    cqcc, audio_fn, tags, labels = [d for d in next(iter(valDataLoader))]
                    cqcc = cqcc.unsqueeze(1).float().to(args.device)
                    tags = tags.to(args.device)
                    labels = labels.to(args.device)
                    feats, cqcc_outputs = cqcc_model(cqcc)

                    score = torch.norm(feats - iso_loss.center, p=2, dim=1)

                    ip1_loader.append(feats)
                    idx_loader.append((labels))
                    tag_loader.append((tags))
                    score_loader.append(score)

                    cqcc_loss = criterion(cqcc_outputs, labels)

                    if args.add_loss in [None, "center", "lgcl"]:
                        devlossDict["feat_loss"].append(cqcc_loss.item())
                        # _, cqcc_predicted = torch.max(cqcc_outputs.data, 1)
                        # total += labels.size(0)
                        # cqcc_correct += (cqcc_predicted == labels).sum().item()
                    elif args.add_loss == "lgm":
                        devlossDict["feat_loss"].append(cqcc_loss.item())
                        # outputs, moutputs, likelihood = lgm_loss(feats, labels)
                        # _, cqcc_predicted = torch.max(outputs.data, 1)
                        # total += labels.size(0)
                        # cqcc_correct += (cqcc_predicted == labels).sum().item()
                    elif args.add_loss == "isolate":
                        isoloss = iso_loss(feats, labels)
                        devlossDict["iso_loss"].append(isoloss.item())
                    elif args.add_loss == "ang_iso":
                        ang_isoloss = ang_iso(feats, labels)
                        devlossDict["ang_iso"].append(ang_isoloss.item())
                    elif args.add_loss == "multi_isolate":
                        multi_isoloss = multi_iso_loss(feats, labels)
                        devlossDict["multi_iso_loss"].append(multi_isoloss.item())
                    elif args.add_loss == "multicenter_isolate":
                        multiisoloss = multicenter_iso_loss(feats, labels)
                        devlossDict["multiiso"].append(multiisoloss.item())
                    # if (k+1) % 10 == 0:
                    #     print('Epoch [{}/{}], Step [{}/{}], cqcc_accuracy {:.4f} %'.format(
                    #         epoch_num + 1, args.num_epochs, k + 1, len(valDataLoader), (100 * cqcc_correct / total)))

                    desc_str = ''
                    for key in sorted(devlossDict.keys()):
                        desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                    v.set_description(desc_str)

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[key])) + "\n")
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[key])) + "\t" + str(eer) +"\n")
            print(eer)

            if args.visualize and ((epoch_num+1) % 3 == 1):
                feat = torch.cat(ip1_loader, 0)
                tags = torch.cat(tag_loader, 0)
                if args.add_loss == "isolate":
                    centers = iso_loss.center
                elif args.add_loss == "multi_isolate":
                    centers = multi_iso_loss.centers
                elif args.add_loss == "ang_iso":
                    centers = ang_iso.center
                else:
                    centers = torch.mean(feat[labels==0], dim=0, keepdim=True)
                visualize(args, feat.data.cpu().numpy(), tags.data.cpu().numpy(), labels.data.cpu().numpy(), centers.data.cpu().numpy(),
                          epoch_num + 1, "Dev")

            valLoss = np.nanmean(devlossDict[key])
            # if args.add_loss == "isolate":
            #     print("isolate center: ", iso_loss.center.data)
            if (epoch_num + 1) % 2 == 0:
                torch.save(cqcc_model, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_cqcc_model_%d.pt' % (epoch_num+1)))
                if args.add_loss == "center":
                    loss_model = centerLoss
                    torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_%d.pt' % (epoch_num+1)))
                elif args.add_loss == "isolate":
                    loss_model = iso_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_%d.pt' % (epoch_num+1)))
                elif args.add_loss == "ang_iso":
                    loss_model = ang_iso
                    torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                        'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
                elif args.add_loss == "multi_isolate":
                    loss_model = multi_iso_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                        'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
                elif args.add_loss == "multicenter_isolate":
                    loss_model = multicenter_iso_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_%d.pt' % (epoch_num+1)))
                elif args.add_loss == "lgm":
                    loss_model = lgm_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_%d.pt' % (epoch_num+1)))
                elif args.add_loss == "lgcl":
                    loss_model = lgcl_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_%d.pt' % (epoch_num+1)))
                else:
                    loss_model = None

            if valLoss < prev_loss:
                # Save the model checkpoint
                torch.save(cqcc_model, os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt'))
                if args.add_loss == "center":
                    loss_model = centerLoss
                    torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                elif args.add_loss == "isolate":
                    loss_model = iso_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                elif args.add_loss == "ang_iso":
                    loss_model = ang_iso
                    torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                elif args.add_loss == "multi_isolate":
                    loss_model = multi_iso_loss
                    torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
                elif args.add_loss == "multicenter_isolate":
                    loss_model = multicenter_iso_loss
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

            if early_stop_cnt == 50:
                with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                    res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 19))
                break
            # if early_stop_cnt == 1:
            #     torch.save(cqcc_model, os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt')

            # print('Dev Accuracy of the model on the val features: {} % '.format(100 * cqcc_correct / total))

    return cqcc_model, loss_model


def test(args, model, loss_model, part='eval'):
    model.eval()
    test_set = ASVspoof2019(args.access_type, args.path_to_database, args.path_to_features, args.path_to_protocol, part,
                            args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size // 2, shuffle=False, num_workers=args.num_workers,
                                collate_fn=test_set.collate_fn)
    cqcc_correct = 0
    total = 0
    ip1_loader, tag_loader, idx_loader = [], [], []
    with open(os.path.join(args.out_fold, 'cm_score.txt'), 'w') as cm_score_file:
    #     for i in trange(len(test_set)):
    #         score_lst = []
    #         for j in range(5):
    #             feat, audio_fn, tags, labels = test_set[i]
    #             feat = feat.unsqueeze(0).unsqueeze(0).float().to(args.device)
    #             feats, _ = model(feat)
    #             # print(torch.norm(feats - loss_model.center, p=2, dim=1).item())
    #             score_lst.append(torch.norm(feats - loss_model.center, p=2, dim=1).item())
    #         score = np.mean(score_lst)
    #         cm_score_file.write('%s A%02d %s %s\n' % (audio_fn, tags,
    #                                               "spoof" if labels else "bonafide",
    #                                               score))

        for i, (cqcc, audio_fn, tags, labels) in enumerate(testDataLoader):
            cqcc = cqcc.unsqueeze(1).float().to(args.device)
            feats, cqcc_outputs = model(cqcc)
            tags = tags.to(args.device)
            labels = labels.to(args.device)
            # ip1_loader.append(feats)
            # tag_loader.append(tags)
            # idx_loader.append(labels)

            if args.add_loss in [None, "center", "isolate", "ang_iso", "multi_isolate", "multicenter_isolate"]:
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
                # print('Test Accuracy of the model on the eval features: {} %'.format(100 * cqcc_correct / total))
            for j in range(labels.size(0)):
                if args.add_loss == "isolate":
                    score = torch.norm(feats[j].unsqueeze(0) - loss_model.center, p=2, dim=1).data.item()
                elif args.add_loss == "ang_iso":
                    score = F.normalize(feats[j].unsqueeze(0), p=2, dim=1) @ F.normalize(loss_model.center, p=2, dim=1).T
                    score = score.data.item()
                elif args.add_loss == "multi_isolate":
                    genuine_dist = torch.norm((feats[j].unsqueeze(0).repeat(args.num_centers, 1) - loss_model.centers), p=2, dim=1)
                    score, indices = torch.min(genuine_dist, dim=-1)
                    score = score.item()
                elif args.add_loss == "multicenter_isolate":
                    score = 1e8
                    for k in range(loss_model.centers.shape[0]):
                        dist = torch.norm(feats[j] - loss_model.centers[k].unsqueeze(0), p=2, dim=1)
                        if dist.item() < score:
                            score = dist.item()
                else:
                    score = cqcc_outputs.data[j][0].cpu().numpy()
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score))
    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(args.out_fold, 'cm_score.txt'), args.path_to_database)

    # feat = torch.cat(ip1_loader, 0)
    # labels = torch.cat(idx_loader, 0)
    # tags = torch.cat(tag_loader, 0)
    # if args.add_loss == "isolate":
    #     centers = loss_model.center
    # elif args.add_loss == "multi_isolate":
    #     centers = loss_model.centers
    # elif args.add_loss == "ang_iso":
    #     centers = loss_model.center
    # else:
    #     centers = torch.mean(feat[labels == 0], dim=0, keepdim=True)
    # torch.save(feat, os.path.join(args.out_fold, 'feat_19.pt'))
    # torch.save(tags, os.path.join(args.out_fold, 'tags_19.pt'))
    # visualize(args, feat.data.cpu().numpy(), tags.data.cpu().numpy(), labels.data.cpu().numpy(), centers.data.cpu().numpy(),
    #           19, part)

    return eer_cm, min_tDCF


if __name__ == "__main__":
    args = initParams()
    if not args.test_only:
        _, _ = train(args)
    model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_cqcc_model.pt'))
    if args.add_loss is None:
        loss_model = None
    else:
        loss_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
    # TReer_cm, TRmin_tDCF = test(args, model, loss_model, "train")
    # VAeer_cm, VAmin_tDCF = test(args, model, loss_model, "dev")
    TEeer_cm, TEmin_tDCF = test(args, model, loss_model)
    with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
        # res_file.write('\nTrain EER: %8.5f min-tDCF: %8.5f\n' % (TReer_cm, TRmin_tDCF))
        # res_file.write('\nVal EER: %8.5f min-tDCF: %8.5f\n' % (VAeer_cm, VAmin_tDCF))
        res_file.write('\nTest EER: %8.5f min-tDCF: %8.5f\n' % (TEeer_cm, TEmin_tDCF))
    plot_loss(args)

    # # Test a checkpoint model
    # args = initParams()
    # model = torch.load(os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_cqcc_model_19.pt'))
    # loss_model = torch.load(os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_19.pt'))
    # VAeer_cm, VAmin_tDCF = test(args, model, loss_model, "dev")
