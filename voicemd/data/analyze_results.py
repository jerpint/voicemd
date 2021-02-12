import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def compute_classification_report(conf_mat):
    preds = []
    labels = []
    tn, fp, fn, tp = conf_mat.ravel()
    #  accuracy = (tn + tp) / (tn + tp + fn + fp)*100

    preds.extend([0]*int(tn))
    labels.extend([0]*int(tn))
    preds.extend([1]*int(tp))
    labels.extend([1]*int(tp))
    preds.extend([1]*int(fp))
    labels.extend([0]*int(fp))
    preds.extend([0]*int(fn))
    labels.extend([1]*int(fn))

    print('='*50)
    print('conf mat: ')
    print(conf_mat)
    print('_'*50)
    print(classification_report(labels, preds))
    print('+'*50)


def report_all_metrics(results_dir, hyper_params):
    '''Aggregate all results after kfold validation.'''
    # TODO: FIX THIS MESS
    categories = hyper_params['categories']
    conf_mat_patients_split = {} # collect for each split
    conf_mat_spectrums_split = {}
    for cat in categories:
        conf_mat_patients_split[f'{cat}'] = []
        conf_mat_spectrums_split[f'{cat}'] = []

    gender_all_confidences = []
    gender_all_targets = []

    total_splits = hyper_params['n_splits']
    for n_splits in range(total_splits):
        with open(os.path.join(results_dir,  'test_results_split_' + str(n_splits) + '.pkl'), 'rb') as f:
            test_results = pickle.load(f)

            # Collect data for ROC curve only for gender
            if 'gender' in categories:
                for patient in test_results['gender_patient_predictions']:
                    if patient['gender_prediction'] == 1:
                        gender_all_confidences.append(patient['gender_confidence'])
                    else:
                        gender_all_confidences.append(1 - patient['gender_confidence'])
                    gender_all_targets.append(patient['gender_gt'])
            for cat in categories:
                conf_mat_patients_split[f'{cat}'].append(test_results[f'{cat}_conf_mat_patients'])
                conf_mat_spectrums_split[f'{cat}'].append(test_results[f'{cat}_conf_mat_spectrums'])

    conf_mat_patients_combined = {}
    conf_mat_spectrums_combined = {}
    for cat in categories:
        conf_mat_patients_combined[f'{cat}'] = sum(conf_mat_patients_split[f'{cat}'])
        conf_mat_spectrums_combined[f'{cat}'] = sum(conf_mat_spectrums_split[f'{cat}'])

        if cat == 'gender':
            print('category: ', cat)
            print('per patient results')
            compute_classification_report(conf_mat_patients_combined[f'{cat}'])
            print('\n'*2)
            print('per spectrum results')
            compute_classification_report(conf_mat_spectrums_combined[f'{cat}'])

    if 'gender' in categories:
        # roc curve only for gender
        fpr, tpr, _ = roc_curve(gender_all_targets, gender_all_confidences)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
        #  plt.show()

if __name__ == "__main__":
    results_dir = 'no_dysphonia_scheduler'
    report_all_metrics(results_dir)
