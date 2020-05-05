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


def get_batch_performance_metrics(outputs, model_target):
    probs = torch.softmax(outputs, 1).detach().numpy() > 0.5
    preds = np.argmax(probs, 1)
    targs = model_target.detach().numpy()
    acc = np.sum(np.equal(preds, targs)) / len(preds)
    conf_mat = confusion_matrix(targs, preds, labels=[0, 1])
    return acc, conf_mat


def performance_metrics_per_patient(patient_predictions):
    patient_targs = []
    patient_preds = []
    for patient_pred in patient_predictions:
        patient_targs.append(patient_pred["gender"])
        patient_preds.append(patient_pred["gender_prediction"])

    conf_mat = confusion_matrix(patient_targs, patient_preds, labels=[0, 1])

    return conf_mat


def evaluate_loaders(loaders, model, loss_fun, device, pb):

    model.eval()
    cumulative_loss = 0.0
    cumulative_acc = 0.0
    cumulative_conf_mat = np.zeros((2, 2))
    patient_predictions = []
    for loader in pb(loaders, total=len(loaders)):
        loader_results = evaluate_loader(loader, model, device, loss_fun)
        cumulative_acc += loader_results["avg_acc"]
        cumulative_loss += loader_results["avg_loss"]
        cumulative_conf_mat += loader_results["conf_mat"]
        patient_predictions.append(loader_results)

    avg_loss = cumulative_loss / len(loaders)
    avg_acc = cumulative_acc / len(loaders)
    per_patient_conf_mat = performance_metrics_per_patient(patient_predictions)

    loaders_results = {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "conf_mat_patients": per_patient_conf_mat,
        "patient_predictions": patient_predictions,
        "conf_mat_spectrums": cumulative_conf_mat,
    }

    return loaders_results


def evaluate_loader(loader, model, device, loss_fun):
    steps = len(loader)
    cumulative_loss = 0.0
    cumulative_acc = 0.0
    cumulative_conf_mat = np.zeros((2, 2))
    examples = 0
    all_probs = []
    for data in loader:
        model_input, model_target = data
        with torch.no_grad():
            outputs = model(model_input.to(device))
            model_target = model_target.type(torch.long)
            model_target = model_target.to(device)
            loss = loss_fun(outputs, model_target)
            cumulative_loss += loss.item()

            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.detach().numpy())
            acc, conf_mat = get_batch_performance_metrics(outputs, model_target)
            cumulative_acc += acc
            cumulative_conf_mat += conf_mat
        examples += model_target.shape[0]

    all_probs = np.array(all_probs)
    avg_prob = np.sum(all_probs, 0) / len(all_probs)
    avg_loss = cumulative_loss / examples
    avg_acc = cumulative_acc / steps
    gender = int(model_target[0])
    final_gender_prediction = np.argmax(avg_prob)
    gender_confidence = avg_prob[final_gender_prediction]
    patient_uid = loader.dataset.metadata.index[0]

    loader_results = {
        "uid": patient_uid,
        "gender": gender,
        "gender_prediction": final_gender_prediction,
        "gender_confidence": gender_confidence,
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "conf_mat": cumulative_conf_mat,
    }

    return loader_results
