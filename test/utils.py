import torch
import numpy as np
import os

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score

from tqdm import tqdm

def normalised_average_precision(y_true, y_pred):

    from sklearn.metrics.ranking import _binary_clf_curve

    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=None,
                                             sample_weight=None)

    n_pos = np.array(y_true).sum()
    n_neg = (1 - np.array(y_true)).sum()

    precision = tps * n_pos / (tps * n_pos + fps * n_neg)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    precision, recall, thresholds = np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def find_nearest(array, value):

    array = np.asarray(array)
    length = len(array)
    abs_diff = np.abs(array - value)

    t_star = abs_diff.min()
    equal_arr = (abs_diff == t_star).astype('float32') + np.linspace(start=0, stop=0.1, num=length)

    idx = equal_arr.argmax()

    return array[idx], idx


def acc_at_t(preds, labels, t):

    pred_t = np.copy(preds)
    pred_t[pred_t > t] = 1
    pred_t[pred_t <= t] = 0

    acc = accuracy_score(labels, pred_t.astype('int32'))

    return acc


def closed_set_acc(preds, labels):

    preds = preds.argmax(axis=-1)
    acc = accuracy_score(labels, preds)

    print('Closed Set Accuracy: {:.3f}'.format(acc))

    return acc


def tar_at_far_and_reverse(fpr, tpr, thresholds):

    # TAR at FAR
    tar_at_far_all = {}
    for t in thresholds:
        tar_at_far_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(fpr, t)
        tar_at_far = tpr[idx]
        tar_at_far_all[t] = tar_at_far

        print(f'TAR @ FAR {t}: {tar_at_far}')

    # FAR at TAR
    far_at_tar_all = {}
    for t in thresholds:
        far_at_tar_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(tpr, t)
        far_at_tar = fpr[idx]
        far_at_tar_all[t] = far_at_tar

        print(f'FAR @ TAR {t}: {far_at_tar}')


def acc_at_95_tpr(open_set_preds, open_set_labels, thresholds, tpr):

    # Error rate at 95% TAR
    _, idx = find_nearest(tpr, 0.95)
    t = thresholds[idx]
    acc_at_95 = acc_at_t(open_set_preds, open_set_labels, t)
    print(f'Error Rate at TPR 95%: {1 - acc_at_95}')

    return acc_at_95


def compute_auroc(open_set_preds, open_set_labels):

    auroc = roc_auc_score(open_set_labels, open_set_preds)
    print(f'AUROC: {auroc}')

    return auroc


def compute_aupr(open_set_preds, open_set_labels, normalised_ap=False):

    if normalised_ap:
        aupr = normalised_average_precision(open_set_labels, open_set_preds)
    else:
        aupr = average_precision_score(open_set_labels, open_set_preds)
    print(f'AUPR: {aupr}')

    return aupr


def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    print(f'OSCR: {OSCR}')

    return OSCR


class EvaluateOpenSet():

    def __init__(self, model, save_dir, known_data_loader, unknown_data_loader, device=None):

        self.model = model
        self.known_data_loader = known_data_loader
        self.unknown_data_loader = unknown_data_loader
        self.save_dir = save_dir

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.device = device

        # Init empty lists for saving labels and preds
        self.closed_set_preds = {0: [], 1: []}
        self.open_set_preds = {0: [], 1: []}

        self.closed_set_labels = {0: [], 1: []}
        self.open_set_labels = {0: [], 1: []}

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def predict(self):

        with torch.no_grad():
            for open_set_label, loader in enumerate((self.known_data_loader, self.unknown_data_loader)):

                if open_set_label:
                    print('Forward pass through Open Set test set...')
                else:
                    print('Forward pass through Closed Set test set...')

                for batch_idx, batch in enumerate(tqdm(loader)):

                    imgs, labels, idxs = [x.to(self.device) for x in batch]

                    # Model forward
                    output = self.model(imgs)
                    closed_set_preds, open_set_preds = [x.cpu().numpy().tolist() for x in output]

                    # Update preds and labels
                    self.closed_set_preds[open_set_label].extend(closed_set_preds)
                    self.open_set_preds[open_set_label].extend(open_set_preds)

                    self.closed_set_labels[open_set_label].extend(labels.cpu().numpy().tolist())
                    self.open_set_labels[open_set_label].extend([open_set_label] * len(labels))

        # Save to disk
        save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']
        save_lists = [self.closed_set_preds, self.open_set_preds, self.closed_set_labels, self.open_set_labels]

        for name, x in zip(save_names, save_lists):

            path = os.path.join(self.save_dir, name)
            torch.save(x, path)

    @staticmethod
    def evaluate(self, load=True, preds=None, normalised_ap=False):

        if load:
            save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = \
                [torch.load(os.path.join(self.save_dir, name)) for name in save_names]

        else:

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = preds


        open_set_preds = np.array(open_set_preds[0] + open_set_preds[1])
        open_set_labels = np.array(open_set_labels[0] + open_set_labels[1])

        # ----------------------------
        # CLOSED SET EVALUATION
        # ----------------------------

        test_acc = closed_set_acc(np.array(closed_set_preds[0]), np.array(closed_set_labels[0]))

        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        fpr, tpr, thresh = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)
        acc_95 = acc_at_95_tpr(open_set_preds, open_set_labels, thresh, tpr)
        auroc = compute_auroc(open_set_preds, open_set_labels)
        aupr = compute_aupr(open_set_preds, open_set_labels, normalised_ap=normalised_ap)

        # OSCR calcs
        open_set_preds_known_cls = open_set_preds[~open_set_labels.astype('bool')]
        open_set_preds_unknown_cls = open_set_preds[open_set_labels.astype('bool')]
        closed_set_preds_pred_cls = np.array(closed_set_preds[0]).argmax(axis=-1)
        labels_known_cls = np.array(closed_set_labels[0])

        oscr = compute_oscr(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls)

        return (test_acc, acc_95, auroc, aupr, oscr)


class EvaluateOpenSetInline(EvaluateOpenSet):

    def __init__(self, *args, **kwargs):

        super(EvaluateOpenSetInline, self).__init__(*args, **kwargs)

    def predict_and_eval(self):

        self.model.eval()

        print('Testing Open Set...')

        with torch.no_grad():
            for open_set_label, loader in enumerate((self.known_data_loader, self.unknown_data_loader)):
                for batch_idx, batch in enumerate(tqdm(loader)):

                    imgs, labels, idxs = [x.to(self.device) for x in batch]

                    # Model forward
                    output = self.model(imgs)
                    closed_set_preds, open_set_preds = [x.cpu().numpy().tolist() for x in output]

                    # Update preds and labels
                    self.closed_set_preds[open_set_label].extend(closed_set_preds)
                    self.open_set_preds[open_set_label].extend(open_set_preds)

                    self.closed_set_labels[open_set_label].extend(labels.cpu().numpy().tolist())
                    self.open_set_labels[open_set_label].extend([open_set_label] * len(labels))

        open_set_preds = np.array(self.open_set_preds[0] + self.open_set_preds[1])
        open_set_labels = np.array(self.open_set_labels[0] + self.open_set_labels[1])

        # ----------------------------
        # CLOSED SET EVALUATION
        # ----------------------------

        test_acc = closed_set_acc(np.array(self.closed_set_preds[0]), np.array(self.closed_set_labels[0]))

        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        fpr, tpr, thresh = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)
        acc_95 = acc_at_95_tpr(open_set_preds, open_set_labels, thresh, tpr)
        auroc = compute_auroc(open_set_preds, open_set_labels)

        return (test_acc, acc_95, auroc)

class ModelTemplate(torch.nn.Module):

    def forward(self, imgs):
        """
        :param imgs:
        :return: Closed set and open set predictions on imgs
        """
        pass

if __name__ == '__main__':

    from sklearn.metrics.ranking import precision_recall_curve

    np.random.seed(0)

    y_true = [0] * 40 + [1] * 60
    y_pred = np.random.uniform(size=(100,))

    def _binary_uninterpolated_average_precision(
            y_true, y_score):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, None, None)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    ap = average_precision_score(y_true, y_pred)
    ap1 = _binary_uninterpolated_average_precision(y_true, y_pred)
    ap2 = normalised_average_precision(y_true, y_pred)

    print(ap)
    print(ap1)
    print(ap2)