import logging
import os

import mlflow
import orion
import yaml
import time
import torch
import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix
from mlflow import log_metric
from orion.client import report_results
from yaml import dump
from yaml import load


def predictions_from_probs(probs):
    probs = np.array(probs)
    collapsed_probs = np.sum(probs, 0) / len(probs)
    class_prediction = np.argmax(collapsed_probs)
    class_confidence = collapsed_probs[class_prediction]
    return class_prediction, class_confidence


def get_batch_performance_metrics(outputs, model_target, labels):
    # gender stats
    probs = torch.softmax(outputs, 1).detach().numpy()
    preds = np.argmax(probs, 1)
    targs = model_target.detach().numpy()
    acc = np.sum(np.equal(preds, targs)) / len(preds)
    conf_mat = confusion_matrix(targs, preds, labels=labels)
    return acc, conf_mat


def performance_metrics_per_patient(patient_predictions, cat, labels):
    patient_targs = []
    patient_preds = []
    for patient_pred in patient_predictions:
        patient_targs.append(patient_pred[f"{cat}_gt"])
        patient_preds.append(patient_pred[f"{cat}_prediction"])

    conf_mat = confusion_matrix(patient_targs, patient_preds, labels=labels)

    return conf_mat


def evaluate_loaders(hyper_params, loaders, model, loss_fun, device, pb):

    combined_results = {}
    for cat in hyper_params['categories']:
        n_cats = len(hyper_params[f'{cat}_label2cat'])
        cumulative_loss = 0.0
        cumulative_acc = 0.0
        per_spectrum_conf_mat = np.zeros((n_cats, n_cats))
        patient_predictions = []
        for loader in pb(loaders, total=len(loaders)):
            loader_results = evaluate_loader(hyper_params, loader, model, device, loss_fun)
            cumulative_acc += loader_results[f"{cat}_avg_acc"]
            cumulative_loss += loader_results[f"{cat}_avg_loss"]
            per_spectrum_conf_mat += loader_results[f"{cat}_conf_mat"]
            patient_predictions.append(loader_results)

        avg_loss = cumulative_loss / len(loaders)
        avg_acc = cumulative_acc / len(loaders)
        labels = list(hyper_params[f'{cat}_label2cat'].values()) # possible label values for cat
        per_patient_conf_mat = performance_metrics_per_patient(patient_predictions, cat, labels)
        combined_results[f"{cat}_avg_loss"] = avg_loss
        combined_results[f"{cat}_avg_acc"] = avg_acc
        combined_results[f"{cat}_conf_mat_patients"] = per_patient_conf_mat
        combined_results[f"{cat}_conf_mat_spectrums"] = per_spectrum_conf_mat
        combined_results[f"{cat}_patient_predictions"] = patient_predictions

    return combined_results


def evaluate_loader(hyper_params, loader, model, device, loss_fun):
    '''Evaluates a single evaluation loader.'''
    from voicemd.train import train_valid_step
    optimizer = None
    with torch.no_grad():
        eval_results = train_valid_step(hyper_params, loader, model, optimizer, loss_fun, device, split='eval')

    return eval_results
