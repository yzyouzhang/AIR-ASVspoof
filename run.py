import os

split_dict = {"1":"23456_1", "2": "13456_2", "3": "12456_3", "4": "12356_4", "5": "12346_5", "6": "12345_6", "0":"123456_"}
for i in range(0, 7):
    # os.system("python train.py --batch_size 64 --add_loss center --feat CQCC --num_epochs 80 --weight_loss 0.00012 -m cnn -p ./traindev_split/"
    #           +split_dict[str(i)]+" -o ./models20200817/crosscnn"+ str(i)+" --gpu 0 --num_workers 0")
    # os.system(
        # "python train_DOC.py --batch_size 64 --add_loss center --feat CQCC --weight_loss 0.0005 --num_epochs 80 -p ./traindev_split/"
        # + split_dict[str(i)] + " -o ./models20200817/doc2_cnn" + str(i) + " --gpu 0 --num_workers 0")
    os.system(
        "python train.py --batch_size 32 --add_loss center --feat Melspec --num_epochs 80 --weight_loss 0.00012 -m resnet -p ./traindev_split/"
        + split_dict[str(i)] + " -o ./models20200824/cross_mel_res" + str(i) + " --gpu 0 --num_workers 0")
# os.system("python train_DOC.py --batch_size 16 --feat Melspec --num_epochs 80 --weight_loss 0.01 --lr 0.00002 -m resnet -o ./models20200824/doc_mel_resnet5 --gpu 1 --num_workers 0")
# os.system("python train_DOC.py --batch_size 16 --feat Melspec --num_epochs 80 --weight_loss 0.001 --lr 0.00002 -m resnet -o ./models20200824/doc_mel_resnet6 --gpu 1 --num_workers 0")
# os.system("python train_DOC.py --batch_size 16 --feat Melspec --num_epochs 80 --weight_loss 0.01 --lr 0.00001 -m resnet -o ./models20200824/doc_mel_resnet7 --gpu 1 --num_workers 0")
# os.system("python train_DOC.py --batch_size 16 --feat Melspec --num_epochs 80 --weight_loss 0.001 --lr 0.00001 -m resnet -o ./models20200824/doc_mel_resnet8 --gpu 1 --num_workers 0")
# os.system("python train_DOC.py --batch_size 16 --feat Melspec --num_epochs 80 --weight_loss 0.0001 --lr 0.00001 -m resnet -o ./models20200824/doc_mel_resnet9 --gpu 1 --num_workers 0")
# os.system("python train_DOC.py --batch_size 16 --feat Melspec --num_epochs 80 --weight_loss 0.1 --lr 0.00001 -m resnet -o ./models20200824/doc_mel_resnet10 --gpu 1 --num_workers 0")
