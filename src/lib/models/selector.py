import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgbm
import logging

# log
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class MSelector:

    """It is a class choosing the optimized pipline using different input parameters"""

    # dictionary for meta learning
    d_param = {
        "class_weight": [0.5, 0.7, 0.9],
        "C": [0.1, 10.0],
        "n_estimator": 300,
        "eta": 0.2,
        "min_child_weight": 2,
        "colsample_bytree": 0.7,
        "scale_pos_weight": [0.5, 0.7]
    }

    d_meta = {
        "count_t": [1200, 2000, 3000],
        "ratio_t": [0, 1, 1/2, 1/3, 1/4, 1/5]
    }


    def __init__(self, cnt, cnt_1, cnt_0):

        self.l_pipe = ["svc_b", "lg_b", "lgbm_b", "svc_i", "lg_i", "lgbm_i"]
        self.cnt = cnt
        self.cnt_1 = cnt_1
        self.cnt_0 = cnt_0
        self.ratio = cnt_1/cnt_0


    @staticmethod
    def switch_pipe(argument, d_param, cnt_1, cnt_0):

        """switcher for getting the optimized clf"""

        def cw(cnt_1, cnt_0, wei0):
            ylen = cnt_1 + cnt_0
            return {0: ((ylen * wei0) / (ylen - cnt_1)),
                    1: (((ylen * (1 - wei0)) / cnt_1))}


        switcher = {
        "svc_b": SVC(
            kernel="linear",
            class_weight=cw(cnt_1, cnt_0, d_param["class_weight"][0]),
            C=d_param["C"][0],
            probability=True,
            max_iter=6000,
            random_state=721
        ),
        "lg_b": LogisticRegression(
            class_weight=d_param["class_weight"][1],
            C=d_param["C"][1],
            tol=0.1,
            n_jobs=-1,
            random_state=831
        ),
        "lgbm_b": lgbm.LGBMClassifier(
            num_leaves=int(6**2*0.8),
            n_estimators=d_param["n_estimator"],
            eta=d_param["eta"],
            min_child_weight=d_param["min_child_weight"],
            colsample_bytree=d_param["colsample_bytree"],
            scale_pos_weight=d_param["scale_pos_weight"][1],
            tree_method="hist",
            max_depth=6,
            random_state=101
        ),
        "svc_i": SVC(
            kernel="linear",
            class_weight=cw(cnt_1, cnt_0, d_param["class_weight"][2]),
            C=d_param["C"][0],
            probability=True,
            max_iter=6000,
            random_state=721
        ),
        "lg_i": LogisticRegression(
            class_weight=d_param["class_weight"][2],
            C=d_param["C"][1],
            tol=0.1,
            n_jobs=-1,
            random_state=831
        ),
        "lgbm_i": lgbm.LGBMClassifier(
            num_leaves=int(6**2*0.8),
            n_estimators=d_param["n_estimator"],
            eta=d_param["eta"],
            min_child_weight=d_param["min_child_weight"],
            colsample_bytree=d_param["colsample_bytree"],
            scale_pos_weight=d_param["scale_pos_weight"][1],
            tree_method="hist",
            max_depth=6,
            random_state=101
        ),
    }
        yield switcher.get(argument, "lg_b")


    @property
    def fit_transform(self):

        # find opt_cnt and opt_ratio
        t_ratio = 3
        opt_cnt = min(self.d_meta["count_t"], key=lambda x: abs(x-self.cnt))
        opt_ratio = min(self.d_meta["ratio_t"], key=lambda x: abs(x-self.ratio))
        logging.info(f"t_ratio: {t_ratio}, opt_cnt: {opt_cnt}, opt_ratio: {opt_ratio}")

        index_cnt = self.d_meta["count_t"].index(opt_cnt)
        index_ratio = self.d_meta["ratio_t"].index(opt_ratio)
        logging.info(f"index_cnt: {index_cnt}, index_ratio: {index_ratio}")

        # filter and get optimized clf
        if index_ratio < t_ratio:
            l_filter_pipe = sorted(list(filter(lambda x: x.find("_b") != -1, self.l_pipe)), key=self.l_pipe.index)
        else:
            l_filter_pipe = sorted(list(filter(lambda x: x.find("_i") != -1, self.l_pipe)), key=self.l_pipe.index)
        argument = l_filter_pipe[index_cnt]
        logging.info(f"f_filter_pipe: {l_filter_pipe}")
        logging.info(f"argument: {argument}")

        clf = next(self.switch_pipe(argument, self.d_param, self.cnt_1, self.cnt_0))
        logging.info(clf)

        return clf
