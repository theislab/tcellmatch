import numpy as np
import sklearn.metrics


def pr_global(
        y_hat,
        y_obs,
        threshold=0.5
):
    """

    :param y_hat:
    :param y_obs:
    :param threshold:
    :return: (data sets, 1) for each metric
    """
    for i in range(len(y_hat)):
        if len(y_hat[i].shape) == 1:
            y_hat[i] = np.expand_dims(y_hat[i], axis=1)
        if len(y_obs[i].shape) == 1:
            y_obs[i] = np.expand_dims(y_obs[i], axis=1)

    is_multiclass = y_obs[0].shape[1] > 1
    if is_multiclass:
        y_hat_disc = [np.zeros_like(y_hat[i]) for i in range(len(y_hat))]
        for i in range(len(y_hat)):
            y_hat_disc[i][np.arange(0, y_hat[i].shape[0]), np.argmax(y_hat[i], axis=1)] = 1
        true_call = [
            np.all((y_obs[i] >= threshold) == (y_hat_disc[i] > 0.5), axis=1)
            for i in range(len(y_hat))
        ]
        tp = [
            np.logical_and(
                true_call[i],
                np.any(y_obs[i][:, :-1] >= threshold, axis=1)
            ).flatten()
            for i in range(len(y_hat_disc))
        ]
        tn = [
            np.logical_and(
                true_call[i],
                y_obs[i][:, -1] >= threshold
            ).flatten()
            for i in range(len(y_hat_disc))
        ]
        fp = [
            np.logical_and(
                np.logical_not(true_call[i]),
                np.any(y_hat_disc[i][:, :-1] >= threshold, axis=1)
            ).flatten()
            for i in range(len(y_hat_disc))
        ]
        fn = [
            np.logical_and(
                np.logical_not(true_call[i]),
                y_hat_disc[i][:, -1] >= threshold
            ).flatten()
            for i in range(len(y_hat_disc))
        ]
    else:
        tp = [
            np.logical_and(y_obs[i] >= threshold, y_hat[i] >= threshold).flatten()
            for i in range(len(y_hat))
        ]
        tn = [
            np.logical_and(y_obs[i] < threshold, y_hat[i] < threshold).flatten()
            for i in range(len(y_hat))
        ]
        fp = [
            np.logical_and(y_obs[i] < threshold, y_hat[i] >= threshold).flatten()
            for i in range(len(y_hat))
        ]
        fn = [
            np.logical_and(y_obs[i] >= threshold, y_hat[i] < threshold).flatten()
            for i in range(len(y_hat))
        ]
    for i, y in enumerate(y_hat):
        assert np.sum(np.sum(np.vstack([tp[i], tn[i], fp[i], fn[i]]).T, axis=1) != 1) == 0, \
            "tp %i, fp %i, tn %i, fn %i, all %i" % \
            (np.sum(tp[i]), np.sum(tn[i]), np.sum(fp[i]), np.sum(fn[i]), y.shape[0])
    tp = np.expand_dims(np.array([np.sum(x) for x in tp]), axis=-1)
    tn = np.expand_dims(np.array([np.sum(x) for x in tn]), axis=-1)
    fp = np.expand_dims(np.array([np.sum(x) for x in fp]), axis=-1)
    fn = np.expand_dims(np.array([np.sum(x) for x in fn]), axis=-1)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    tpr = tp / np.expand_dims(np.array([x.shape[0] for x in y_obs]), axis=-1)
    tnr = tn / np.expand_dims(np.array([x.shape[0] for x in y_obs]), axis=-1)
    fpr = fp / np.expand_dims(np.array([x.shape[0] for x in y_obs]), axis=-1)
    fnr = fn / np.expand_dims(np.array([x.shape[0] for x in y_obs]), axis=-1)
    return precision, recall, tpr, tnr, fpr, fnr


def pr_label(
    y_hat,
    y_obs,
    labels,
    labels_unique,
    threshold=0.5
):
    """

    :param y_hat:
    :param y_obs:
    :param labels:
    :param threshold:
    :return: (data sets, labels_unique) for each metric
    """
    assert len(y_hat) == len(y_obs)
    assert len(y_hat) == len(labels)
    assert np.all([y_hat[i].shape == y_obs[i].shape for i in range(len(y_hat))])
    for i in range(len(y_hat)):
        if len(y_hat[i].shape) == 1:
            y_hat[i] = np.expand_dims(y_hat[i], axis=1)
        if len(y_obs[i].shape) == 1:
            y_obs[i] = np.expand_dims(y_obs[i], axis=1)
        if labels[i] is not None:
            if len(labels[i]) == 0:
                labels[i] = None
    for i in range(len(y_hat)):
        if labels[i] is not None:
            assert y_hat[i].shape[0] == len(labels[i]), \
                "%i, %i \n %s" % (y_hat[i].shape[0], len(labels[i]), str(labels[i]))
    is_multiclass = y_obs[0].shape[1] > 1
    if is_multiclass:
        y_hat_bool = [np.ones_like(y_hat[i]) == 0 for i in range(len(y_hat))]
        for i in range(len(y_hat)):
            for j in range(y_hat[i].shape[0]):
                y_hat_bool[i][j, np.argmax(y_hat[i][j, :])] = True
    else:
        y_hat_bool = [x >= threshold for x in y_hat]
    tp = [
        np.logical_and(y_obs[i] >= threshold, y_hat_bool[i])
        for i in range(len(y_obs))
    ]
    tn = [
        np.logical_and(y_obs[i] < threshold, np.logical_not(y_hat_bool[i]))
        for i in range(len(y_obs))
    ]
    fp = [
        np.logical_and(y_obs[i] < threshold, y_hat_bool[i])
        for i in range(len(y_obs))
    ]
    fn = [
        np.logical_and(y_obs[i] >= threshold, np.logical_not(y_hat_bool[i]))
        for i in range(len(y_obs))
    ]
    if labels[0] is None or y_obs[0].shape[1] > 1:
        # labels are grouped in columns of y and confusion table arrays.
        tp = np.concatenate([np.sum(x, axis=0, keepdims=True) for x in tp], axis=0)
        tn = np.concatenate([np.sum(x, axis=0, keepdims=True) for x in tn], axis=0)
        fp = np.concatenate([np.sum(x, axis=0, keepdims=True) for x in fp], axis=0)
        fn = np.concatenate([np.sum(x, axis=0, keepdims=True) for x in fn], axis=0)
    elif labels[0] is not None and y_obs[0].shape[1] == 1:
        assert labels_unique is not None, "supply labels_unique"
        # y and confusion table arrays all have a single column and labels correspond to sets of rows.
        tp = np.concatenate([
            np.concatenate([
                np.sum(x[labels[i] == y, :], axis=0, keepdims=True)
                if np.sum(labels[i] == y) > 0 else np.array([[np.nan]])
                for y in labels_unique
            ], axis=-1)
            for i, x in enumerate(tp)
        ], axis=0)
        tn = np.concatenate([
            np.concatenate([
                np.sum(x[labels[i] == y, :], axis=0, keepdims=True)
                if np.sum(labels[i] == y) > 0 else np.array([[np.nan]])
                for y in labels_unique
            ], axis=-1)
            for i, x in enumerate(tn)
        ], axis=0)
        fp = np.concatenate([
            np.concatenate([
                np.sum(x[labels[i] == y, :], axis=0, keepdims=True)
                if np.sum(labels[i] == y) > 0 else np.array([[np.nan]])
                for y in labels_unique
            ], axis=-1)
            for i, x in enumerate(fp)
        ], axis=0)
        fn = np.concatenate([
            np.concatenate([
                np.sum(x[labels[i] == y, :], axis=0, keepdims=True)
                if np.sum(labels[i] == y) > 0 else np.array([[np.nan]])
                for y in labels_unique
            ], axis=-1)
            for i, x in enumerate(fn)
        ], axis=0)
    else:
        assert False
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return precision, recall, tp, tn, fp, fn


def auc_global(
    y_hat,
    y_obs
):
    """

    :param y_hat:
    :param y_obs:
    :return: (data sets, 1)
    """
    for i in range(len(y_hat)):
        if len(y_hat[i].shape) == 1:
            y_hat[i] = np.expand_dims(y_hat[i], axis=1)
        if len(y_obs[i].shape) == 1:
            y_obs[i] = np.expand_dims(y_obs[i], axis=1)
    auc_roc = np.expand_dims(np.array([
        sklearn.metrics.roc_auc_score(
            y_true=y_obs[i],
            y_score=y_hat[i]
        ) if np.all([len(np.unique(y_obs[i][:, j])) > 1 for j in range(y_obs[0].shape[1])]) else np.nan
        for i in range(len(y_hat))
    ]), axis=-1)
    return auc_roc


def auc_label(
    y_hat: list,
    y_obs: list,
    labels: list,
    labels_unique: list
):
    """

    :param y_hat:
    :param y_obs:
    :param labels:
    :return: (data sets, labels_unique)
    """
    assert len(y_hat) == len(y_obs)
    assert len(y_hat) == len(labels)
    assert np.all([y_hat[i].shape == y_obs[i].shape for i in range(len(y_hat))])
    for i in range(len(y_hat)):
        if len(y_hat[i].shape) == 1:
            y_hat[i] = np.expand_dims(y_hat[i], axis=1)
        if len(y_obs[i].shape) == 1:
            y_obs[i] = np.expand_dims(y_obs[i], axis=1)
        if labels[i] is not None:
            if len(labels[i]) == 0:
                labels[i] = None
    for i in range(len(y_hat)):
        if labels[i] is not None:
            assert y_hat[i].shape[0] == len(labels[i]), \
                "%i, %i \n %s" % (y_hat[i].shape[0], len(labels[i]), str(labels[i]))
    if labels[0] is None or y_obs[0].shape[1] > 1:
        auc_roc = np.concatenate([
            np.expand_dims(np.array([
                sklearn.metrics.roc_auc_score(
                    y_true=y_obs[i][:, j],
                    y_score=y_hat[i][:, j]
                ) if len(np.unique(y_obs[i][:, j])) > 1 else np.nan
                for j in range(y_hat[i].shape[1])
            ]), axis=0)
            for i in range(len(y_hat))
        ], axis=0)
    elif labels[0] is not None and y_obs[0].shape[1] == 1:
        assert labels_unique is not None, "supply labels_unique"
        auc_roc = np.concatenate(np.expand_dims(np.array([
            np.concatenate(np.expand_dims(np.array([
                sklearn.metrics.roc_auc_score(
                    y_true=y_obs[i][labels[i] == label, 0],
                    y_score=y_hat[i][labels[i] == label, 0],
                ) if len(np.unique(y_obs[i][labels[i] == label, 0])) > 1 else np.nan
                for label in labels_unique
            ]), axis=-1), axis=-1)
            for i in range(len(y_hat))
        ]), axis=0), axis=0)
    else:
        assert False
    return auc_roc


def deviation_global(
        y_hat,
        y_obs
):
    """
    Global deviation metrics: Correlation and error

    :param y_hat:
    :param y_obs:
    :param threshold:
    :return: (data sets, 1) for each metric
    """
    for i in range(len(y_hat)):
        if len(y_hat[i].shape) == 1:
            y_hat[i] = np.expand_dims(y_hat[i], axis=1)
        if len(y_obs[i].shape) == 1:
            y_obs[i] = np.expand_dims(y_obs[i], axis=1)

    mse = np.concatenate([
          np.expand_dims(np.mean(np.square(y_obs[i] - y_hat[i])), axis=0)
          for i in range(len(y_obs))
    ], axis=0)
    msle = np.concatenate([
        np.expand_dims(np.mean(np.square(np.log(y_obs[i]+1) - np.log(y_hat[i]+1))), axis=0)
        for i in range(len(y_obs))
    ], axis=0)
    r2 = np.concatenate([
        np.expand_dims(
            1 - np.sum(np.square(y_obs[i] - y_hat[i])) / \
            np.sum(np.square(y_obs[i] - np.mean(y_obs[i]))),
            axis=0
        )
        for i in range(len(y_obs))
    ], axis=0)
    r2log = np.concatenate([
        np.expand_dims(
            1 - np.sum(np.square(np.log(y_obs[i]+1) - np.log(y_hat[i]+1))) / \
            np.sum(np.square(np.log(y_obs[i]+1) - np.mean(np.log(y_obs[i]+1)))),
            axis=0
        )
        for i in range(len(y_obs))
    ], axis=0)

    return mse, msle, r2, r2log


def deviation_label(
        y_hat,
        y_obs,
        labels: list,
        labels_unique: list
):
    """
    Local deviation metrics: Correlation and error

    :param y_hat:
    :param y_obs:
    :param threshold:
    :return: (data sets, labels_unique) for each metric
    """
    assert len(y_hat) == len(y_obs)
    assert len(y_hat) == len(labels)
    assert np.all([y_hat[i].shape == y_obs[i].shape for i in range(len(y_hat))])
    for i in range(len(y_hat)):
        if len(y_hat[i].shape) == 1:
            y_hat[i] = np.expand_dims(y_hat[i], axis=1)
        if len(y_obs[i].shape) == 1:
            y_obs[i] = np.expand_dims(y_obs[i], axis=1)
        if labels[i] is not None:
            if len(labels[i]) == 0:
                labels[i] = None
    for i in range(len(y_hat)):
        if labels[i] is not None:
            assert y_hat[i].shape[0] == len(labels[i]), \
                "%i, %i \n %s" % (y_hat[i].shape[0], len(labels[i]), str(labels[i]))

    if labels[0] is None or y_obs[0].shape[1] > 1:
        # labels are grouped in columns of y and confusion table arrays.
        mse = np.concatenate([
            np.expand_dims(np.mean(np.square(y_obs[i] - y_hat[i]), axis=0), axis=0)
            for i in range(len(y_obs))
        ], axis=0)
        msle = np.concatenate([
            np.expand_dims(np.mean(np.square(np.log(y_obs[i] + 1) - np.log(y_hat[i] + 1)), axis=0), axis=0)
            for i in range(len(y_obs))
        ], axis=0)
        r2 = np.concatenate([
            np.expand_dims(
                1 - np.sum(np.square(y_obs[i] - y_hat[i]), axis=0) / \
                np.sum(np.square(y_obs[i] - np.mean(y_obs[i], axis=0, keepdims=True)), axis=0),
                axis=0
            )
            for i in range(len(y_obs))
        ], axis=0)
        r2log = np.concatenate([
            np.expand_dims(
                1 - np.sum(np.square(np.log(y_obs[i] + 1) - np.log(y_hat[i] + 1)), axis=0) / \
                np.sum(np.square(np.log(y_obs[i] + 1) - np.mean(np.log(y_obs[i] + 1), keepdims=True, axis=0)), axis=0),
                axis=0
            )
            for i in range(len(y_obs))
        ], axis=0)
    elif labels[0] is not None and y_obs[0].shape[1] == 1:
        assert labels_unique is not None, "supply labels_unique"
        # y and confusion table arrays all have a single column and labels correspond to sets of rows.
        mse = np.concatenate([
            np.expand_dims(
                np.array([
                    np.mean(np.square(y_obs[i] - y_hat[i][labels[i] == y, :]))
                    if np.sum(labels[i] == y) > 0 else np.nan
                    for y in labels_unique
                ]),
                axis=0
            )
            for i in range(len(y_obs))
        ], axis=0)
        msle = np.concatenate([
            np.expand_dims(
                np.array([
                    np.mean(np.square(
                        np.log(y_obs[i][labels[i] == y, :] + 1) -
                        np.log(y_hat[i][labels[i] == y, :] + 1)
                    )) if np.sum(labels[i] == y) > 0 else np.nan
                    for y in labels_unique
                ]),
                axis=0
            )
            for i in range(len(y_obs))
        ], axis=0)
        r2 = np.concatenate([
            np.expand_dims(
                np.array([
                    1 - np.sum(np.square(
                        y_obs[i][labels[i] == y, :] -
                        y_hat[i][labels[i] == y, :]))
                    / np.sum(np.square(
                        y_obs[i][labels[i] == y, :] -
                        np.mean(y_obs[i][labels[i] == y, :])
                    )) if np.sum(labels[i] == y) > 0 else np.nan
                    for y in labels_unique
                ]),
                axis=0
            )
            for i in range(len(y_obs))
        ], axis=0)
        r2log = np.concatenate([
            np.expand_dims(
                np.array([
                    1 - np.sum(np.square(
                        np.log(y_obs[i][labels[i] == y, :] + 1) -
                        np.log(y_hat[i][labels[i] == y, :] + 1)
                    )) / np.sum(np.square(
                        np.log(y_obs[i][labels[i] == y, :] + 1) -
                        np.mean(np.log(y_obs[i][labels[i] == y, :] + 1))
                    )) if np.sum(labels[i] == y) > 0 else np.nan
                    for y in labels_unique
                ]),
                axis=0
            )
            for i in range(len(y_obs))
        ], axis=0)
    else:
        assert False

    return mse, msle, r2, r2log
