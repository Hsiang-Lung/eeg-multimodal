import torch
from torch.utils.data import DataLoader

from framework.dataset import EEGDataset, EEG_Table_Dataset
from framework.model import Model, train, test
from framework import utils
import warnings
import os
import numpy as np

warnings.filterwarnings('ignore')


def run_training(data_path, binary, transform_train, device, run_id, configs, split_count, fold='', folds=0):
    # if the result folders do not exist, then they are created
    path = os.getcwd() + '/framework/training_results/plots/' + run_id
    if os.path.exists(path):
        print("training_results already exist for this run_id and data will be overwriten in result folder")
    else:
        os.mkdir(path)
        os.mkdir(os.getcwd() + '/framework/training_results/checkpoints/' + run_id)
        for i in range(folds):
            os.mkdir(path + '/fold' + str(i))
            os.mkdir(os.getcwd() + '/framework/training_results/checkpoints/' + run_id + '/fold' + str(i))

    # select the correct datasettype, depending on some configs
    if configs['table'] and not binary:
        train_dataset = EEG_Table_Dataset(path=data_path + 'train', transform=transform_train, 
            columnDropProb=configs['column_dropout'], columnDropIdx=configs['column_dropout_idx'])
        val_dataset = EEG_Table_Dataset(path=data_path + 'test')
    else:
        train_dataset = EEGDataset(path=data_path + 'train', transform=transform_train)
        val_dataset = EEGDataset(path=data_path + 'test')

    # create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=split_count, shuffle=False, num_workers=4, pin_memory=True)

    # some variables
    statsLoss, statsAcc = [], []
    dataHistory, dataHistoryPatient = [], []
    num_classes = (2 if binary else 3)
    sequence_len = train_dataset[0][0].shape[1]
    model = Model(num_classes=num_classes, sequence_len=sequence_len, with_table=(configs['table'] if not binary else False)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    best_val_loss, best_val_age = 100.0, 0

    # calculate weights for weighted loss if toggled
    if configs['weighted_loss']:
        all_labels = train_dataset.labels
        nSamples = [torch.sum(all_labels == i).item() for i in range(num_classes)]
        weights = [1 - (x / sum(nSamples)) for x in nSamples]
        weights = torch.FloatTensor(weights).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    # run training for n epochs
    for epoch in range(configs['epochs']):
        print(f"Epoch {epoch + 1}:")
        t_acc, t_loss = train(train_loader, model, criterion, optimizer, device)
        v_acc, v_loss, all_y, all_pred = test(val_loader, model, criterion, device, forPatient=False)
        v_accP, v_lossP, all_yP, all_predP = test(val_loader, model, criterion, device, forPatient=True)
        dataHistory.append((t_acc, t_loss, v_acc, v_loss))
        dataHistoryPatient.append((t_acc, t_loss, v_accP, v_lossP))

        # if a new lowest val loss is achieved, we save the current (best) model 
        if best_val_loss > v_loss:
            best_val_loss = v_loss
            best_val_age = 0
            torch.save(model.state_dict(),
                       os.getcwd() + '/framework/training_results/checkpoints/' + run_id+fold + '/best_model.pt')
            print('new lowest val loss: {}\n'.format(v_loss))
        else:
            print('no new lowest val loss; patience: {}/{}\n'.format(best_val_age, configs['patience']))
            best_val_age += 1
            # check if a certain patience threshold is reached, then test a last time and stop training
            if best_val_age > configs['patience']:
                v_acc, v_loss, all_y, all_pred = test(val_loader, model, criterion, device, forPatient=False)
                v_accP, v_lossP, all_yP, all_predP = test(val_loader, model, criterion, device, forPatient=True)
                break

    # save the training/val history in a csv
    utils.save_epoch_history_csv(run_id, configs['epochs'], dataHistory, run_id + fold)

    # plot the acc/loss history 
    statsAcc.append(("TrainAcc", [i[0] for i in dataHistory]))
    statsAcc.append(("ValAcc", [i[2] for i in dataHistory]))
    statsAcc.append(("TrainAcc_Patient", [i[0] for i in dataHistoryPatient]))
    statsAcc.append(("ValAcc_Patient", [i[2] for i in dataHistoryPatient]))
    statsLoss.append(("TrainLoss", [i[1] for i in dataHistory]))
    statsLoss.append(("ValLoss", [i[3] for i in dataHistory]))
    statsLoss.append(("TrainLoss_Patient", [i[1] for i in dataHistoryPatient]))
    statsLoss.append(("ValLoss_Patient", [i[3] for i in dataHistoryPatient]))
    utils.plot_history(run_id + "_Accuracy", statsAcc, run_id + fold, isAcc=True)
    utils.plot_history(run_id + "_Loss", statsLoss, run_id + fold, isAcc=False)
    
    # plot ROC and confusion matrix
    if binary:
        pass
        # TODO future work: fix ploting 
        #utils.plot_roc(run_id + "ROC", all_y, all_pred, run_id+fold)
        #utils.plot_roc(run_id + "ROC_Patient", all_yP, all_predP, run_id+fold)
    else:
        utils.plot_roc_multi(run_id + "ROC", all_y, all_pred, run_id+fold)
        utils.plot_roc_multi(run_id + "ROC_Patient", all_yP, all_predP, run_id+fold)
        utils.plot_confusionmatrix(run_id + "CM", all_y, all_pred, run_id + fold)
        utils.plot_confusionmatrix(run_id + "CM_Patient", all_yP, all_predP, run_id + fold)
    return all_yP, all_predP
