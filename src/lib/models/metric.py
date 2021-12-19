import numpy as np
import logging

# log
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class Metrics:

    """It is a class calculating precision and recall
       with y_true and y_pred in np array. The model comparison
       will use f1-score as the model metric. User can choose
       whether using macro f1-score or weighted f1-score."""


    def __init__(self, y_true, y_pred):

        self.y_true = y_true
        self.y_pred = y_pred

    @property
    def confusion_matrix(self):

        """To compute the confusion matrix from y_true and y_pred"""
        logging.info("confusion_matrix")

        ul_ = len(np.unique(self.y_true))
        cm = np.zeros((ul_, ul_))

        logging.info("loop and create confusion_matrix")
        for i in range(len(self.y_true)):
            cm[self.y_true[i]][self.y_pred[i]] += 1

        return cm

    @staticmethod
    def fit(cm):

        """To calculate tp, fp and fn"""
        logging.info("calculate actual_sum, predict_sum and tp_sum")

        actual_sum = np.sum(cm, axis=1)
        predict_sum = np.sum(cm, axis=0)
        tp_sum = cm.diagonal()

        logging.info("calculate precision and recall")
        precision = tp_sum / predict_sum
        recall = tp_sum / actual_sum

        return precision, recall

    @staticmethod
    def macro_precision_recall(y_true, precision, recall):

        """calculate macro_precision_recall"""
        logging.info("macro_precision_recall")

        num_y = len(np.unique(y_true))

        macro_precision = np.sum(precision) / num_y
        macro_recall = np.sum(recall) / num_y

        return macro_precision, macro_recall

    @staticmethod
    def weighted_precision_recall(y_true, precision, recall):

        """calculate weighted_precision_recall"""
        logging.info("weighted_precision_recall")

        total_y = len(y_true)
        unique_y, count_y = np.unique(y_true, return_counts=True)

        weighted_precision = np.sum(precision * count_y) /  total_y
        weighted_recall = np.sum(recall * count_y) / total_y

        return weighted_precision, weighted_recall

    @staticmethod
    def f1(y_true, precision, recall, flag="macro"):

        """calculate macro_f1"""
        logging.info("macro_f1")

        try:
            f1_score = 2 * (precision * recall) / (precision + recall)
        except Exception as e:
            f1_score = 0

        if flag == "macro":
            num_y = len(np.unique(y_true))
            return np.sum(f1_score) / num_y
        else:
            total_y = len(y_true)
            unique_y, count_y = np.unique(y_true, return_counts=True)
            return np.sum(f1_score * count_y) / total_y