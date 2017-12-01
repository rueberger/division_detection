""" Contains methods for result analysis
"""

import os
import h5py
import numpy as np

from division_detection.utils import clean_dir
from division_detection.model import fetch_model
from division_detection.vol_preprocessing import pipeline_batch_generator, fetch_validation_annotations
from division_detection.sparse_utils import load_coo_as_dense

from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from skimage.io import imsave

from six import iteritems

from warnings import warn

def select_examplars(test_data, test_labels, test_paths, predicted_labels, model_name,
                     n_examplars=250, save=False):
    """ Selects a few interesting examples to plot
    The most confidently misclassified, and the closest to the decision boundary
    Writes the examplars as tiffs, also returns results for good measure

    Args:
       model: trained keras model
       test_data: test data - arr
       test_labels: test labels - arr
       test_paths: paths to full res - list
       model_name: vgg or conv
       n_examplars: number of examplars from each class, twice this total
    """
    raw_predictions = predicted_labels
    bin_predictions = (raw_predictions > 0.5).astype(int)
    # distance from 0.5
    confidences = np.abs(raw_predictions - 0.5)

    misclassified = bin_predictions != test_labels
    misclass_idxs = np.where(misclassified)[0]

    # global idxs of most confidently misclassified examples
    high_conf_misclass_idxs = misclass_idxs[np.argsort(confidences[misclassified])[-n_examplars:]]

    mcm_confs = np.sort(confidences[misclassified])[-n_examplars:]

    print("Confidences for most confident misclassifications: {}".format(mcm_confs))

    most_confused_idxs = np.argsort(confidences)[:n_examplars]
    print("Least confident confidences: {}".format(confidences[most_confused_idxs].squeeze()))

    results = {
        'most_confused': test_data[most_confused_idxs],
        'most_confused_labels': test_labels[most_confused_idxs],
        'confused_confidences': mcm_confs,
        'most_confident_misclass': test_data[high_conf_misclass_idxs],
        'most_confident_misclass_labels': test_labels[high_conf_misclass_idxs],
        'mcm_confidences': confidences[high_conf_misclass_idxs]
    }

    if save:
        print("Saving examplars...")
        result_dir = os.path.expanduser('~/results/div_detect/examplars/{}'.format(model_name))
        clean_dir(result_dir)

        examplar_set = set()
        for hcm_idx in high_conf_misclass_idxs:
            full_res_path = test_paths[hcm_idx]
            pred_label = bin_predictions[hcm_idx]
            true_label = test_labels[hcm_idx]
            confidence = confidences[hcm_idx]

            link_path = '{}/pred_{}_true_{}_conf_{}.klb'.format(
                result_dir, pred_label, true_label, confidence)
            if full_res_path not in examplar_set:
                os.symlink(full_res_path, link_path)
                examplar_set.add(full_res_path)



        # for conf_idx in most_confused_idxs:
        #     conf = confidences[conf_idx][0]
        #     # the center t slice only
        #     vol =  test_data[conf_idx, :, :, :, 1]
        #     imsave('{}/confused_{}_{}.tif'.format(result_dir, conf_idx, conf), vol, plugin='tifffile')

    return results

def pipeline_analyze(model_name, partials=True, test=True):
    """ Pipelined analysis method
    """
    model, _ = fetch_model(model_name)

    if partials:
        from division_detection.vol_preprocessing import SPLIT_PARTIALS_PATH_TEMPLATE, REC_FIELD_SHAPE
        if test:
            partials_path = SPLIT_PARTIALS_PATH_TEMPLATE.format('test')
        else:
            partials_path = SPLIT_PARTIALS_PATH_TEMPLATE.format('train')

        with h5py.File(partials_path, 'r') as partials_file:
            # [n_samples] + REC_FIELD_SHAPE
            partial_cutouts = partials_file[str(tuple(REC_FIELD_SHAPE))]['cutouts'][:]
            # [n_samples,]
            labels = partials_file[str(tuple(REC_FIELD_SHAPE))]['labels'][:]

        raw_predictions = model.predict(partial_cutouts).squeeze()

    # evaluate PR on fully annotated validation volumes
    else:
        annotations = fetch_validation_annotations()
        valid_tps = np.unique(annotations[:, 0]).astype(np.int32)

        prediction_path = '/nrs/turaga/bergera/division_detection/prediction_outbox/{}.h5'.format(model_name)
        gt_path = os.path.expanduser('~/data/div_detect/full_res_gt_vols/validation.h5')

        if not os.path.exists(prediction_path):
            raise RuntimeError("Predictions file missing")

        # flattened over all volumes
        raw_predictions = []
        labels = []
        with h5py.File(prediction_path) as predictions_file, h5py.File(gt_path) as gt_file:
            predictions = predictions_file['predictions']
            for timept in valid_tps:
                tp_predict = predictions[timept]
                gt_vol = gt_file[str(timept)][:]
                if tp_predict.sum() > 0:
                    raw_predictions.append(tp_predict.ravel())
                    labels.append(gt_vol.ravel())
                else:
                    warn("No predictions found for timepoint {}".format(timept))

        if len(raw_predictions) == 0:
            raise RuntimeError("No validation predictions found")

        raw_predictions = np.concatenate(raw_predictions)
        labels = np.concatenate(labels)

    class_predictions = (raw_predictions > 0.5).astype(int)
    correct_predictions = class_predictions == labels
    test_accuracy = np.sum(correct_predictions) / float(len(correct_predictions))
    n_pos_samples = np.sum(labels)
    n_neg_samples = np.sum(np.logical_not(labels))

    print("Achieved {} test set accuracy".format(test_accuracy))
    print("Test set contains {} positive examples and {} negative examples".format(
        n_pos_samples,
        n_neg_samples
    ))

    print("Computing precision recall curve")
    precision, recall, thresholds = precision_recall_curve(labels.ravel(), raw_predictions.ravel(), pos_label=1)
    precision_recall_dict = {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }

    print("Computing ROC curve")
    false_pos_rate, true_pos_rate, thresholds = roc_curve(labels.ravel(), raw_predictions.ravel(), pos_label=1)
    roc_dict = {
        'false_pos_rate': false_pos_rate,
        'true_pos_rate': true_pos_rate,
        'thresholds': thresholds
    }

    print('Computing confusion matrix')
    decision_thresholds = [0.1, 0.3, 0.5, 0.9, 0.95]
    confusion_matrices = {thresh: confusion_matrix(labels.ravel(), raw_predictions.ravel() > thresh) for thresh in decision_thresholds}

    analysis_results = {
        'pr_curve': precision_recall_dict,
        'roc_curve': roc_dict,
        'confusion_matrices': confusion_matrices
    }

    for thresh, cm in iteritems(confusion_matrices):
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix at decision threshold of {}:".format(thresh))
        print(norm_cm)

    return analysis_results

def pipeline_visualize(model_name, plus_minus=1):
    """ Saves max projections and possibly other visualizations for the predictions for each model
    """
    from division_detection.vol_preprocessing import VAL_TIMEPOINTS, FULL_RES_GT_TARGET_PATH, fetch_bb_at_t
    from division_detection.plotting import save_img
    from division_detection.vol_preprocessing import fetch_annotations
    train_an = fetch_annotations('train')
    test_an = fetch_annotations('test')
    val_an = fetch_annotations('validation')
    used_tps = np.unique(np.concatenate([train_an, test_an, val_an])[:, 0]).astype(int)


    save_path = os.path.expanduser('~/results/div_detect/visualizations/{}'.format(model_name))
    pred_path_template = '/nrs/turaga/bergera/division_detection/prediction_outbox/{0}/sparse/'.format(model_name)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print("Here we go")

    with h5py.File(FULL_RES_GT_TARGET_PATH, 'r') as gt_file:
        for mid_tp in used_tps:
            os.mkdir('{}/{}'.format(save_path, mid_tp))
            bbox = fetch_bb_at_t(mid_tp)
            gt_vol = gt_file[str(mid_tp)]
            gt_vol_crop = gt_vol[bbox[0][0]: bbox[0][1],
                                 bbox[1][0]: bbox[1][1],
                                 bbox[2][0]: bbox[2][1]]
            gt_crop_max_proj = np.max(gt_vol_crop, axis=0)
            gt_full_max_proj = np.max(gt_vol, axis=0)
            save_img(gt_crop_max_proj, '{}/{}/gt_tp_{}_cropped'.format(save_path, mid_tp, mid_tp))
            save_img(gt_full_max_proj, '{}/{}/gt_tp_{}_full'.format(save_path, mid_tp, mid_tp))

            projs = []
            for val_tp in range(mid_tp - plus_minus, mid_tp + plus_minus + 1):
                try:
                    print("Loading tp {}".format(val_tp))
                    bbox = fetch_bb_at_t(val_tp)
                    pred_path = pred_path_template + '{}.h5'.format(val_tp)
                    prediction = load_coo_as_dense(pred_path)
                    prediction_crop = prediction[bbox[0][0]: bbox[0][1],
                                                 bbox[1][0]: bbox[1][1],
                                                 bbox[2][0]: bbox[2][1]]
                    pred_full_max_proj = np.max(prediction, axis=0)
                    pred_crop_max_proj = np.max(prediction_crop, axis=0)

                    projs.append(pred_full_max_proj)
                    save_img(pred_full_max_proj, '{}/{}/pred_tp_{}_full'.format(save_path, mid_tp, val_tp))
                    save_img(pred_crop_max_proj, '{}/{}/pred_tp_{}_cropped'.format(save_path, mid_tp, val_tp))

                except OSError as os_err:
                    print("Caught OSError on tp {}: {}".format(val_tp, os_err))

            if len(projs) > 0:
                combined_pred = np.sum(projs, axis=0)
                save_img(combined_pred, '{}/{}/combined_pred_pm_{}'.format(save_path, mid_tp, plus_minus))

                img = np.stack([projs[1], gt_full_max_proj], axis=-1)
                save_img(img, '{}/{}/combined_tryptich_pm_{}'.format(save_path, mid_tp, plus_minus))

                img = np.stack([projs[1], gt_full_max_proj], axis=-1)
                save_img(img, '{}/{}/single_tp_tryptich'.format(save_path, mid_tp))

def analyze_performance(model, model_name, get_examplars=True, include_validation=False):
    """ Dispatch method to perform all standard analysis

    Args:
      model: a trained keras model object
      model_name: (optional) vgg or conv.
      get_examplars: (optional) if true, return examplars

    Returns:
      precision_recall_dict
      roc_dict
      activations
      examplar_dict: (optional)
    """
    print("Loading test data")
    data_dir = os.path.expanduser('~/data/div_detect')
    with h5py.File('{}/test_recs.h5'.format(data_dir), 'r') as test_file:
        test_data = test_file['records'][:]
        test_labels = test_file['labels'][:]
        test_paths = list(test_file['record_paths'][:])

    if include_validation:
        print("Loading validation data as well")
        with h5py.File('{}/valid_recs.h5'.format(data_dir), 'r') as valid_file:
            test_data = np.append(test_data, valid_file['records'][:], axis=0)
            test_labels = np.append(test_labels, valid_file['labels'][:], axis=0)
            test_paths += list(valid_file['record_paths'][:])

    print("Computing predicted labels")
    raw_predictions = model.predict(test_data, verbose=1)
    class_predictions = (raw_predictions > 0.5).astype(int)
    correct_predictions = class_predictions == test_labels
    test_accuracy = np.sum(correct_predictions) / float(len(correct_predictions))
    n_pos_samples = np.sum(test_labels)
    n_neg_samples = np.sum(np.logical_not(test_labels))

    print("Achieved {} test set accuracy".format(test_accuracy))
    print("Test set contains {} positive examples and {} negative examples".format(
        n_pos_samples,
        n_neg_samples
    ))

    print("Computing precision recall curve")
    precision, recall, thresholds = precision_recall_curve(test_labels.ravel(), raw_predictions.ravel(), pos_label=1)
    precision_recall_dict = {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }

    print("Computing ROC curve")
    false_pos_rate, true_pos_rate, thresholds = roc_curve(test_labels.ravel(), raw_predictions.ravel(), pos_label=1)
    roc_dict = {
        'false_pos_rate': false_pos_rate,
        'true_pos_rate': true_pos_rate,
        'thresholds': thresholds
    }

    if get_examplars:
        print("Selecting examplars")
        examplar_dict = select_examplars(test_data, test_labels, test_paths,
                                         raw_predictions, model_name, save=True)
        return precision_recall_dict, roc_dict, raw_predictions, examplar_dict
    else:
        return precision_recall_dict, roc_dict, raw_predictions
