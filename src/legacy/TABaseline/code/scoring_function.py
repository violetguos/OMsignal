import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import recall_score


def scorePerformance(prMean_pred, prMean_true, rtMean_pred, rtMean_true,
                     rrStd_pred, rrStd_true, ecgId_pred, ecgId_true):
    """
    Computes the combined multitask performance score. The 3 regression tasks
    are individually scored using Kendalls correlation coefficient. the user
    classification task is scored according to macro averaged recall, with an
    adjustment for chance level. All performances are clipped at 0.0, so that
    zero indicates chance or worse performance, and 1.0 indicates perfect
    performance. The individual performances are then combined by taking the
    geometric mean.

    :param prMean_pred: 1D float32 numpy array. The predicted average P-R
     interval duration over the window. One row for each window.
    :param prMean_true: 1D float32 numpy array. The true average P-R interval
     duration over the window. One row for each window.
    :param rtMean_pred: 1D float32 numpy array. The predicted average R-T
     interval duration over the window. One row for each window.
    :param rtMean_true: 1D float32 numpy array. The true average R-T interval
     duration over the window. One row for each window.
    :param rrStd_pred: 1D float32 numpy array. The predicted R-R interval
     duration standard deviation over the window. One row for each window.
    :param rrStd_true: 1D float32 numpy array. The true R-R interval duration
     standard deviation over the window. One row for each window.
    :param ecgId_pred: 1D int32 numpy array. containing the predicted user ID
     label for each window.
    :param ecgId_true: 1D int32 numpy array. containing the true user ID label
     for each window.
    :return: The combined performance score on all tasks; 0.0 means at least
     one task has chance level performance or worse, 1.0 means all tasks are
     solved perfectly.
    The individual task performance scores are also returned
    """

    numElmts = None

    # Input checking
    if ecgId_true is not None:
        assert isinstance(ecgId_pred, np.ndarray)
        assert len(ecgId_pred.shape) == 1
        assert ecgId_pred.dtype == np.int32

        assert isinstance(ecgId_true, np.ndarray)
        assert len(ecgId_true.shape) == 1
        assert ecgId_true.dtype == np.int32

        assert (len(ecgId_pred) == len(ecgId_true))
        if numElmts is not None:
            assert (len(ecgId_pred) == numElmts) and (
                len(ecgId_true) == numElmts)
        else:
            numElmts = len(ecgId_pred)

    if rrStd_true is not None:
        assert isinstance(rrStd_pred, np.ndarray)
        assert len(rrStd_pred.shape) == 1
        assert rrStd_pred.dtype == np.float32

        assert isinstance(rrStd_true, np.ndarray)
        assert len(rrStd_true.shape) == 1
        assert rrStd_true.dtype == np.float32

        assert (len(rrStd_pred) == len(rrStd_true))
        if numElmts is not None:
            assert (len(rrStd_pred) == numElmts) and (
                len(rrStd_true) == numElmts)
        else:
            numElmts = len(rrStd_pred)

    if prMean_true is not None:
        assert isinstance(prMean_pred, np.ndarray)
        assert len(prMean_pred.shape) == 1
        assert prMean_pred.dtype == np.float32

        assert isinstance(prMean_true, np.ndarray)
        assert len(prMean_true.shape) == 1
        assert prMean_true.dtype == np.float32

        assert (len(prMean_pred) == len(prMean_true))
        if numElmts is not None:
            assert (len(prMean_pred) == numElmts) and (
                len(prMean_true) == numElmts)
        else:
            numElmts = len(prMean_pred)

    if rtMean_true is not None:
        assert isinstance(rtMean_pred, np.ndarray)
        assert len(rtMean_pred.shape) == 1
        assert rtMean_pred.dtype == np.float32

        assert isinstance(rtMean_true, np.ndarray)
        assert len(rtMean_true.shape) == 1
        assert rtMean_true.dtype == np.float32

        assert (len(rtMean_pred) == len(rtMean_true))
        if numElmts is not None:
            assert (len(rtMean_pred) == numElmts) and (
                len(rtMean_true) == numElmts)
        else:
            numElmts = len(rtMean_pred)

    if numElmts is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    numVal = 0.0

    # Accuracy is computed with macro averaged recall so that accuracy is
    # computed as though the classes were balanced, even if they are not. Note
    # that the training, validation and testing sets are balanced as given.
    # Unbalanced classes would only be and issue if a new train/validation
    # split is created.
    # Any accuracy value worse than random chance will be clipped at zero.
    if ecgId_true is not None:
        numVal += 1.0
        ecgIdAccuracy = recall_score(ecgId_true, ecgId_pred, average='macro')
        adjustementTerm = 1.0 / len(np.unique(ecgId_true))
        ecgIdAccuracy = (ecgIdAccuracy - adjustementTerm) / \
            (1 - adjustementTerm)
        if ecgIdAccuracy < 0:
            ecgIdAccuracy = 0.0
        ecgIdAccuracyRep = ecgIdAccuracy

    else:
        ecgIdAccuracy = 1.0
        ecgIdAccuracyRep = 0.0

    # Compute Kendall correlation coefficients for regression tasks.
    # Any coefficients worse than chance will be clipped to zero.
    if rrStd_true is not None:
        numVal += 1.0
        rrStdTau, _ = kendalltau(rrStd_pred, rrStd_true)
        if rrStdTau < 0:
            rrStdTau = 0.0
        rrStdTauRep = rrStdTau
    else:
        rrStdTau = 1.0
        rrStdTauRep = 0.0

    if prMean_true is not None:
        numVal += 1.0
        prMeanTau, _ = kendalltau(prMean_pred, prMean_true)
        if prMeanTau < 0:
            prMeanTau = 0.0
        prMeanTauRep = prMeanTau
    else:
        prMeanTau = 1.0
        prMeanTauRep = 0.0

    if rtMean_true is not None:
        numVal += 1.0
        rtMeanTau, _ = kendalltau(rtMean_pred, rtMean_true)
        if rtMeanTau < 0:
            rtMeanTau = 0.0
        rtMeanTauRep = rtMeanTau
    else:
        rtMeanTau = 1.0
        rtMeanTauRep = 0.0

    # Compute the final performance score as the geometric mean of the
    # individual task performances.
    # A high geometric mean ensures that there are no tasks with very poor
    # performance that are masked by good performance on the other tasks.
    # If any task has chance performance or worse, the overall performance will
    # be zero. If all tasks are perfectly solved, the overall performance will
    # be 1.
    combinedPerformanceScore = np.power(
        rrStdTau * prMeanTau * rtMeanTau * ecgIdAccuracy,
        1.0 / max(1.0, numVal)
    )

    return (
        combinedPerformanceScore,
        prMeanTauRep,
        rtMeanTauRep,
        rrStdTauRep,
        ecgIdAccuracyRep
    )


def example():
    prMean_pred = np.random.randn(480).astype(np.float32)
    prMean_true = (np.random.randn(480).astype(
        np.float32) / 10.0) + prMean_pred

    rtMean_pred = np.random.randn(480).astype(np.float32)
    rtMean_true = (np.random.randn(480).astype(
        np.float32) / 10.0) + rtMean_pred

    rrStd_pred = np.random.randn(480).astype(np.float32)
    rrStd_true = (np.random.randn(480).astype(np.float32) / 10.0) + rrStd_pred

    ecgId_pred = np.random.randint(low=0, high=32, size=(480,), dtype=np.int32)
    ecgId_true = np.random.randint(low=0, high=32, size=(480,), dtype=np.int32)

    print(
        scorePerformance(
            prMean_true, prMean_pred, rtMean_true, rtMean_pred, rrStd_true,
            rrStd_pred, ecgId_true, ecgId_pred
        )
    )
