""" This module contains plotting utils
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import PIL.Image
from io import BytesIO
import IPython.display


import seaborn as sns
sns.set_style('white')


def vis_vol_as_slices(vol, title, slice_axis=0, cmap='veridis'):
    """ visualize the volume as slices along the specified axis
    """
    assert vol.ndim == 3

    n_slices = vol.shape[slice_axis]

    slices = np.split(vol, n_slices, axis=slice_axis)
    spacer = np.zeros((len(slices), 5)) * np.nan
    img_sando = []
    for img_slice in slices:
        img_sando.append(img_slice.squeeze())
        img_sando.append(spacer[:])
    img_sando = img_sando[:-1]
    stacked_img = np.concatenate(img_sando, axis=1)
    fig = plt.figure(figsize=(16, 16))
    ax = plt.gca()
    img = ax.imshow(stacked_img, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    return fig

def vis_tvol_as_slices(tvol, title, slice_axis=0, cmap='gray'):
    """ Visualize a 4d vol as slices
    """
    assert tvol.ndim == 4

    n_vols = tvol.shape[-1]
    n_slices = tvol.shape[slice_axis]

    vol_sando = []
    vols = np.split(tvol, n_vols, axis=-1)
    for vol in vols:
        slice_sando = []
        slices = np.split(vol, n_slices, axis=slice_axis)
        spacer = np.zeros((len(slices), 5)) * np.nan

        for vol_slice in slices:
            slice_sando.append(vol_slice.squeeze())
            slice_sando.append(spacer[:])

        slice_sando = slice_sando[:-1]
        slice_img = np.concatenate(slice_sando, axis=1)

        vol_sando.append(slice_img)
        vol_sando.append(np.zeros((5, slice_img.shape[1])) * np.nan)

    full_img = np.concatenate(vol_sando, axis=0)
    fig = plt.figure(figsize=(16, 16))
    ax = plt.gca()
    img = ax.imshow(full_img, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    return fig

def plot_perf_curve(curve_data, curve_type, model_names,
                    axis=None, tight_layout=True):
    """ Convenience method for plotting performance curves

    Args:
      curve_data: curve dictionary or list of curve dictionaries
      curve_type: 'pr' or 'roc'
      model_names: name of model that made each curve - list(str)
      axis: (optional) axis to plot too. if not passed, a new figure is created

    Returns:
      axis: the axis plot was made on

    """
    if isinstance(curve_data, dict):
        curve_data = [curve_data]
    if isinstance(model_names, str):
        model_names = [model_names]

    assert len(curve_data) == len(model_names)

    if axis is None:
        fig = plt.Figure()
        axis = plt.gca()

    if curve_type == 'pr':
        for curve_dict, model_name in zip(curve_data, model_names):
            axis.plot(curve_dict['precision'], curve_dict['recall'], label=model_name)
            axis.set_xlabel('Precision')
            axis.set_ylabel('Recall')
            if tight_layout:
                xlim_idx = np.where(np.isclose(curve_dict['recall'], 1, atol=1e-3))[0][-1]
                xlim = curve_dict['precision'][xlim_idx]
                ylim_idx = np.where(np.isclose(curve_dict['precision'], 1, atol=1e-3))[0][0]
                ylim = curve_dict['recall'][ylim_idx]
                axis.set_xlim([xlim, 1])
                axis.set_ylim([ylim, 1])
            else:
                axis.set_xlim([0, 1])
                axis.set_ylim([0, 1])
            axis.grid(True)
        axis.legend()
    elif curve_type == 'roc':
        for curve_dict, model_name in zip(curve_data, model_names):
            axis.plot(curve_dict['false_pos_rate'], curve_dict['true_pos_rate'], label=model_name)
            axis.set_xlabel('False positive rate')
            axis.set_ylabel('True positive rate')
            if tight_layout:
                xlim_idx = np.where(np.isclose(curve_dict['true_pos_rate'], 1, atol=1e-3))[0][0]
                xlim = curve_dict['false_pos_rate'][xlim_idx]
                ylim_idx = np.where(np.isclose(curve_dict['false_pos_rate'], 0, atol=1e-3))[0][-1]
                ylim = curve_dict['true_pos_rate'][ylim_idx]
                axis.set_xlim([0, xlim])
                axis.set_ylim([0, ylim])
            else:
                axis.set_xlim([0, 1])
                axis.set_ylim([0, 1])
            axis.grid(True)
        axis.legend()
    return axis

def plot_joint_perf_curves(curve_data, model_names):
    """ Plots ROC and PR side by side

    Args:
      curve_data: a tuple (pr_curve, roc_curve) or list of tupels [(pr_curve, roc_curve)]
      model_name: name of model that made each curves - list(str)

    Returns:
      fig: the finished figure
    """
    assert isinstance(curve_data, dict)
    if isinstance(model_names, str):
        model_names = [model_names]

    fig, ((pr_ax), (roc_ax)) = plt.subplots(1, 2, figsize=(12, 4))
    pr_curve_data = curve_data['pr_curve']
    roc_curve_data = curve_data['roc_curve']

    pr_ax = plot_perf_curve(pr_curve_data, 'pr', model_names, axis=pr_ax)
    roc_ax = plot_perf_curve(roc_curve_data, 'roc', model_names, axis=roc_ax)

    fig.set_canvas(plt.gcf().canvas)
    return fig

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, ax=None):
    """
    Adapted from: http://scikit-learn.org/Stable/Auto_Examples/Model_Selection/Plot_Confusion_Matrix.Html#Sphx-Glr-Auto-Examples-Model-Selection-Plot-Confusion-Matrix-Py

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    if ax is None:
        fig = plt.Figure()
        ax = plt.gca()

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def pipeline_plot_confusion_matrix(results_rec):
    """  Pipelined confusion matrix plot
    """
    from six import iteritems

    fig, axes = plt.subplots(len(results_rec['confusion_matrices']), 1)
    for axis, (thresh, c_matrix) in zip(axes, iteritems(results_rec['confusion_matrices'])):
        plot_confusion_matrix(c_matrix, classes=['negative', 'positive'], title='Confusion matrix at threshold of {}'.format(thresh), normalize=True, ax=axis)

def save_img(img, save_path, fmt='png', norm=True):
    """ Render and save img
    """
    if norm:
        if np.max(img) != 0:
           img = np.uint8(((img - np.min(img)) / np.max(img)) * 255)
        else:
            img = img

    with open('{}.{}'.format(save_path, fmt), 'wb') as  img_file:
        pil_img = PIL.Image.fromarray(img)
        if pil_img.mode != 'RGB':
            pil_img.convert('RGB')
        pil_img.save(img_file, fmt)


def show_img(img, fmt='png'):
    if np.max(img) != 0:
        a = np.uint8(((img - np.min(img)) / np.max(img)) * 255)
    else:
        a = img
    f = BytesIO()
    pil_img = PIL.Image.fromarray(a)
    if pil_img.mode != 'RGB':
        pil_img.convert('RGB')
    pil_img.save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
