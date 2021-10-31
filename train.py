import argparse
import os
import json
import shutil
from resnet import setup_seed, ResNet
from loss import *
from dataset import ASVspoof2019
from collections import defaultdict
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import torch
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/dataNVME/neil/ASVspoof2019LAFeatures/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=598)

    parser.add_argument('--add_loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for ocsoftmax")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for ocsoftmax")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for ocsoftmax")

    parser.add_argument('--continue_training', action='store_true', help="continue training with previously trained model")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        assert os.path.exists(args.out_fold)
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
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    # assign device
    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    lfcc_model = ResNet(3, args.enc_dim, resnet_type='18', nclasses=2).to(args.device)
    if args.continue_training:
        lfcc_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt')).to(args.device)

    lfcc_optimizer = torch.optim.Adam(lfcc_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'train',
                                'LFCC', feat_len=args.feat_len, padding=args.padding)
    validation_set = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'dev',
                                  'LFCC', feat_len=args.feat_len, padding=args.padding)
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=training_set.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               collate_fn=validation_set.collate_fn)

    feat, _, _, _ = training_set[29]
    print("Feature shape", feat.shape)

    criterion = nn.CrossEntropyLoss()

    if args.add_loss == "amsoftmax":
        amsoftmax_loss = AMSoftmax(2, args.enc_dim, s=args.alpha, m=args.r_real).to(args.device)
        amsoftmax_loss.train()
        amsoftmax_optimzer = torch.optim.SGD(amsoftmax_loss.parameters(), lr=0.01)

    if args.add_loss == "ocsoftmax":
        ocsoftmax = OCSoftmax(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha).to(args.device)
        ocsoftmax.train()
        ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=args.lr)

    early_stop_cnt = 0
    prev_eer = 1e8

    monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        lfcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        adjust_learning_rate(args, lfcc_optimizer, epoch_num)
        if args.add_loss == "ocsoftmax":
            adjust_learning_rate(args, ocsoftmax_optimzer, epoch_num)
        elif args.add_loss == "amsoftmax":
            adjust_learning_rate(args, amsoftmax_optimzer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(trainDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(args.device)
            labels = labels.to(args.device)
            feats, lfcc_outputs = lfcc_model(lfcc)
            lfcc_loss = criterion(lfcc_outputs, labels)

            if args.add_loss == "softmax":
                lfcc_optimizer.zero_grad()
                trainlossDict[args.add_loss].append(lfcc_loss.item())
                lfcc_loss.backward()
                lfcc_optimizer.step()

            if args.add_loss == "ocsoftmax":
                ocsoftmaxloss, _ = ocsoftmax(feats, labels)
                lfcc_loss = ocsoftmaxloss * args.weight_loss
                lfcc_optimizer.zero_grad()
                ocsoftmax_optimzer.zero_grad()
                trainlossDict[args.add_loss].append(ocsoftmaxloss.item())
                lfcc_loss.backward()
                lfcc_optimizer.step()
                ocsoftmax_optimzer.step()

            if args.add_loss == "amsoftmax":
                outputs, moutputs = amsoftmax_loss(feats, labels)
                lfcc_loss = criterion(moutputs, labels)
                trainlossDict[args.add_loss].append(lfcc_loss.item())
                lfcc_optimizer.zero_grad()
                amsoftmax_optimzer.zero_grad()
                lfcc_loss.backward()
                lfcc_optimizer.step()
                amsoftmax_optimzer.step()

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(np.nanmean(trainlossDict[monitor_loss])) + "\n")

        # Val the model
        lfcc_model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(valDataLoader)):
                lfcc = lfcc.unsqueeze(1).float().to(args.device)
                labels = labels.to(args.device)

                feats, lfcc_outputs = lfcc_model(lfcc)

                lfcc_loss = criterion(lfcc_outputs, labels)
                score = F.softmax(lfcc_outputs, dim=1)[:, 0]

                if args.add_loss == "softmax":
                    devlossDict["softmax"].append(lfcc_loss.item())
                elif args.add_loss == "amsoftmax":
                    outputs, moutputs = amsoftmax_loss(feats, labels)
                    lfcc_loss = criterion(moutputs, labels)
                    score = F.softmax(outputs, dim=1)[:, 0]
                    devlossDict[args.add_loss].append(lfcc_loss.item())
                elif args.add_loss == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(feats, labels)
                    devlossDict[args.add_loss].append(ocsoftmaxloss.item())
                idx_loader.append(labels)
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[monitor_loss])) + "\t" + str(val_eer) +"\n")
            print("Val EER: {}".format(val_eer))

        torch.save(lfcc_model, os.path.join(args.out_fold, 'checkpoint',
                                            'anti-spoofing_lfcc_model_%d.pt' % (epoch_num + 1)))
        if args.add_loss == "ocsoftmax":
            loss_model = ocsoftmax
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
        elif args.add_loss == "amsoftmax":
            loss_model = amsoftmax_loss
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
        else:
            loss_model = None

        if val_eer < prev_eer:
            # Save the model checkpoint
            torch.save(lfcc_model, os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))
            if args.add_loss == "ocsoftmax":
                loss_model = ocsoftmax
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            elif args.add_loss == "amsoftmax":
                loss_model = amsoftmax_loss
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            else:
                loss_model = None
            prev_eer = val_eer
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 19))
            break

    return lfcc_model, loss_model


if __name__ == "__main__":
    args = initParams()
    _, _ = train(args)
    model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))
    if args.add_loss == "softmax":
        loss_model = None
    else:
        loss_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
