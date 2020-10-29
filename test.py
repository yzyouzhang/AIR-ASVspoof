import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_model(feat_model_path, loss_model_path, part, add_loss):
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
                            "LFCC", feat_len=750, padding="repeat")
    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        score_loader = []
        for i, (cqcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            cqcc = cqcc.unsqueeze(1).float().to(device)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, cqcc_outputs = model(cqcc)

            score = cqcc_outputs[:, 0]

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'),
                                            "/data/neil/DS_10283_3336/")
    return eer_cm, min_tDCF

def test(model_dir, add_loss):
    model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    test_model(feat_model_path, loss_model_path, "eval", add_loss)

if __name__ == "__main__":
    model_dir = "/data/neil/antiRes/models1028/softmax"
    test(model_dir, None)

    model_dir = "/data/neil/antiRes/models1028/amsoftmax"
    test(model_dir, "amsoftmax")

    model_dir = "/data/neil/antiRes/models1028/ocsoftmax"
    test(model_dir, "ocsoftmax")

