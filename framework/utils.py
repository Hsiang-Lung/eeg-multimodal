import csv
import os
from itertools import cycle

import matplotlib.pyplot as plt
import torch
import torchvision as tv
import pandas as pd
import numpy as np
import random
from mne import filter
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import auc
from pathlib import Path
from yasa import sliding_window
from framework.augmentations import augmentation as aug
import torch.fft as fft
import mne


def save_epoch_history_csv(filename, epoch, dataHistory, run_id):
    # write the result data to a csv file
    dir = os.getcwd()
    with open(dir + "/framework/training_results/plots/" + run_id + '/' + filename + ".csv", "w",
              newline="") as f:
        writer = csv.writer(f)
        epoch_number = []
        for i in range(epoch + 1):
            epoch_number.append(i)
        writer.writerow(["Epoch"] + list(map(lambda x: x, epoch_number)))
        writer.writerow(["Acc_Train"] + list(map(lambda x: x, [round(i[0], 4) for i in dataHistory])))
        writer.writerow(["Acc_Val"] + list(map(lambda x: x, [round(i[2], 4) for i in dataHistory])))
        writer.writerow(["Loss_Train"] + list(map(lambda x: x, [round(i[1], 6) for i in dataHistory])))
        writer.writerow(["Loss_Val"] + list(map(lambda x: x, [round(i[3], 6) for i in dataHistory])))


def plot_history(filename, stats, run_id, isAcc=True):
    # save plot of acc/loss
    dir = os.getcwd()
    for i in stats:
        plt.plot(i[1], label=i[0])
    plt.legend(loc="upper left")
    plt.title(filename)
    plt.xlabel('epoch')
    plt.ylabel('accuracy') if isAcc else plt.ylabel('loss')
    plt.savefig(dir + "/framework/training_results/plots/" + run_id + '/' + filename + '.png')
    plt.close()


def samplesplit_subsampling(data, labels, num, table=None):
    # splits the EEG singal into num signals, by subsampling
    split_data1 = [data[:, :, i::num] for i in range(num)]
    patient_num = len(split_data1[0])
    split_data2 = []
    for i in range(patient_num):
        for y in range(len(split_data1)):
            split_data2.append(torch.unsqueeze(split_data1[y][i], 0))

    if not split_data2[0].size() == split_data2[-1].size():
        size = split_data2[-1].size()[2]
        for idx in range(len(split_data2)):
            split_data2[idx] = split_data2[idx][:, :, :size]

    data = np.concatenate(split_data2, axis=0)
    labels = np.repeat(labels, num)
    table = np.repeat(table, num, 0)
    return data, labels, table


def samplesplit_timeframe(data, labels, table=None, freq=200, window=5, step=4):
    # segments EEG signal based on the given window and step sizes
    split_data = []
    split_count = 0
    for idx, sample in enumerate(data):
        x = sliding_window(data[idx], sf=freq, window=window, step=step)[1]
        split_data.append(x)
        if idx == 0:
            split_count = x.shape[0]

    split_data = np.concatenate(split_data, axis=0)
    labels = np.repeat(labels, split_count)
    table = np.repeat(table, split_count, axis=0)
    split_num = len(split_data) / len(data)
    return split_data, labels, table, split_num


def normalization(data, norm_type='channel'):
    # zero-mean normalization
    norm_data = None
    if norm_type == 'dataset':
        mean = np.mean(data.flatten())
        std = np.std(data.flatten())
        norm_data = torch.tensor((data - mean) / std)
    elif norm_type == 'channel':
        mean = np.mean(data, axis=2)[:, :, np.newaxis]
        std = np.std(data, axis=2)[:, :, np.newaxis]
        norm_data = torch.tensor((data - mean) / std)
    elif norm_type == 'sample':
        mean = np.mean(data, axis=(1, 2))[:, np.newaxis, np.newaxis]
        std = np.std(data, axis=(1, 2))[:, np.newaxis, np.newaxis]
        print(std.shape)
        norm_data = torch.tensor((data - mean) / std)
    return norm_data


def plot_roc(filename, y, y_pred, run_id):
    # plots the ROC based on the models predictions
    dir = os.getcwd()

    y_pred = np.array(y_pred).argmax(axis=-1).tolist()  # ADDED

    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    model_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b-', label='AUC = {:2.2f}'.format(model_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(fontsize=12)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC curve')
    plt.savefig(dir + "/framework/training_results/plots/" + run_id + '/' + filename + '.png')
    plt.close()


def plot_roc_crossval(y_list, y_pred_list, run_id):
    # plots the ROC curves for each crossval fold and the mean ROC curve
    # based on the models predictions

    target_names = ['chronic_bp', 'fibro', 'various']
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for class_id, color in zip(range(3), colors):
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, y in enumerate(y_list):
            label_binarizer = LabelBinarizer().fit(y)
            y_onehot_test = label_binarizer.transform(y)
            y_pred = y_pred_list[i]
            y_pred = np.array(y_pred)
            viz = RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_pred[:, class_id],
                name=f"ROC curve for {target_names[class_id]}",
                color=color,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        ax.plot_loss([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot_loss(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n(Positive label '{target_names[class_id]}')",
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        Path(os.getcwd() + "/framework/training_results/plots/" + run_id).mkdir(exist_ok=True)
        plt.savefig(os.getcwd() + "/framework/training_results/plots/" + run_id + "/mean_ROC" + target_names[
            class_id] + ".png")
        plt.close()


def plot_roc_multi(filename, y, y_pred, run_id):
    # plots the ROC based on the models predictions (for multiclass)
    y_pred = np.array(y_pred)
    target_names = ['chronic_bp', 'fibro', 'various']
    label_binarizer = LabelBinarizer().fit(y)
    y_onehot_test = label_binarizer.transform(y)
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'r--')

    for class_id, color in zip(range(3), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_pred[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
        )
    plt.savefig(os.getcwd() + "/framework/training_results/plots/" + run_id + '/' + filename + '.png')
    plt.close()


def plot_confusionmatrix(filename, y, y_pred, run_id):
    # plots the confusion matrix for multiclass based on predictions
    cm = confusion_matrix(y, np.array(y_pred).argmax(axis=1), labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['chronic_bp', 'fibro', 'various'])
    disp.plot()
    plt.savefig(os.getcwd() + "/framework/training_results/plots/" + run_id + '/' + filename + '.png')
    plt.close()


def get_augmentations(c):
    # Composes the transform augmentations based on the information in the config file

    augs = []
    if c['channel_dropout']['is']:
        augs.append(aug.ChannelDropout(c['time']['prob']))
    if c['ftsurr']['is']:
        augs.append(aug.FourierTransformSurrogate(phase_noise_magnitude=c['ftsurr']['noise'],
                                                  channel_indep=c['ftsurr']['indep'], prob=c['ftsurr']['prob']))
    if c['freqshift']['is']:
        augs.append(aug.FrequencyShift(delta_freq=c['freqshift']['delta'], sfreq=c['freqshift']['sfreq'],
                                       prob=c['freqshift']['prob']))
    if c['rescale']['is']:
        augs.append(
            aug.RandomRescale(mu=c['rescale']['mu'], sigma=c['rescale']['sigma'], channel_prob=c['rescale']['prob']))
    if c['gauss']['is']:
        augs.append(aug.GaussianNoise(mu=c['gauss']['mu'], sigma=c['gauss']['sigma'], channel_prob=c['gauss']['prob']))
    if c['sign']['is']:
        augs.append(aug.SignFlip(channel_prob=c['sign']['prob']))
    if c['time']['is']:
        augs.append(aug.TimeFlip(channel_prob=c['time']['prob']))

    return tv.transforms.Compose(augs)


def notch_filter(eeg_data, fs, notch_freq):
    # Applies a notch filter with a two-sided margin of 1Hz to the signal

    eeg_data = torch.from_numpy(eeg_data)
    N = eeg_data.shape[-1]
    Y = fft.fftshift(fft.fft(eeg_data, dim=-1), dim=-1)

    center = int(0.5 * N)
    offset = int(notch_freq * (N / fs))
    band = int(1 * (N / fs))

    Y[:, :, center - offset - band:center - offset + band] = 0
    Y[:, :, center + offset - band:center + offset + band] = 0

    eeg_data = fft.ifft(fft.ifftshift(Y, dim=-1), dim=-1)
    eeg_data = torch.real(eeg_data)

    return eeg_data.numpy()


def bandpass_filter(eeg_data, fs, freq_min, freq_max):
    # Apply bandpass filter to the signal

    eeg_data = torch.from_numpy(eeg_data)
    nb_channels = eeg_data.shape[1]
    ch_types = ['eeg'] * nb_channels
    info = mne.create_info(nb_channels, sfreq=fs, ch_types=ch_types)
    all_patients = mne.EpochsArray(eeg_data, info)
    all_patients.filter(freq_min, freq_max)
    eeg_data = torch.tensor(np.array(all_patients.get_data(), dtype='float32'))
    return eeg_data.numpy()


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def calc_splitcount(config):
    # calculates the amount of samples the original sample is split into
    count = 1
    if 'subsampling' in config['sample_split']:
        count = int(config['subsampling_count'])

    if 'segments' in config['sample_split']:
        segment_step = config['segment_step']
        segment_window = config['segment_window']
        c = 275
        ctr = 0
        while c >= segment_window:
            c -= segment_step
            ctr += 1
        count *= ctr
        # count *= int(275 / segment_step)

    return count


# hardcoded method to create the necessary folder for training
def createFolders():
    # if the result folders do not exist, then they are created
    path = os.getcwd() + '/framework/data_splits'
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        os.mkdir(path + '/multiclass_split')
        os.mkdir(path + '/multiclass_split/train')
        os.mkdir(path + '/multiclass_split/test')
        os.mkdir(os.getcwd() + '/framework/training_results')
        os.mkdir(os.getcwd() + '/framework/training_results/plots')
        os.mkdir(os.getcwd() + '/framework/training_results/checkpoints')