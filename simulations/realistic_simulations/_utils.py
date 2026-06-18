import numpy as np


def roc_curve(y_true, score):
    """Compute (fpr, tpr) by sweeping thresholds over `score`, descending.

    A sample is called positive when score >= threshold, so larger `score`
    means "more likely positive".
    """
    order = np.argsort(-score)
    y_true = np.asarray(y_true)[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    tps = np.cumsum(y_true)
    fps = np.cumsum(~y_true)

    tpr = np.concatenate([[0], tps / n_pos])
    fpr = np.concatenate([[0], fps / n_neg])

    return fpr, tpr


def roc_auc_score(y_true, score):
    fpr, tpr = roc_curve(y_true, score)
    return np.trapz(tpr, fpr)


def set_size(width, fraction=1, subplots=(3, 3)):
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
