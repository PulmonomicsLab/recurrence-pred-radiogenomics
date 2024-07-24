from typing import List
from .model import Model
import numpy as np

svmParamGridRBF = {
    'kernel': ['rbf'],
    'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    'C': [1, 5, 10]
}

svmParamGridPoly = {
    'kernel': ['poly'],
    'degree': [2, 3],
    'gamma': [1e-1, 1e-2, 1e-3],
    'coef0': [0, 1, 2, 3],
    'C': [1, 5, 10]
}

svmParamGridLinear = {
    'kernel': ['linear'],
    'C': [1, 5, 10, 15]
}

rfParamGrid = {
    'n_estimators': [200, 400, 600, 800, 1000],
    'max_depth': [None],
    'max_features': [0.25, 0.5, 0.75, None]
}

# mlpParamGrid = {
#     'activation': ['relu'],
#     'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
#     'hidden_layer_sizes':
#         []
#         + [(50,), (100,), (150,), (200,), (250,), (300,), (350,), (400,), (450,), (500,), (550,), (600,), (650,), (700,), (750,), (800,), (850,), (900,), (950,), (1000,)]
#         + [(50, 10), (100, 20), (150, 30), (200, 40), (250, 50), (300, 60), (350, 70), (400, 80), (450, 90), (500, 100), (550, 110), (600, 120), (650, 130), (700, 140), (750, 150), (800, 160), (850, 170), (900, 180), (950, 290), (1000, 200)]
#         + [(50, 25), (100, 50), (150, 75), (200, 100), (250, 125), (300, 150), (350, 175), (400, 200), (450, 225), (500, 250), (550, 275), (600, 300), (650, 325), (700, 350), (750, 375), (800, 400), (850, 425), (900, 450), (950, 475), (1000, 500)]
#         + [(50, 50), (100, 100), (150, 150), (200, 200), (250, 250), (300, 300), (350, 350), (400, 400), (450, 450), (500, 500), (550, 550), (600, 600), (650, 650), (700, 700), (750, 750), (800, 800), (850, 850), (900, 900), (950, 950), (1000, 1000)]
# }

mlpParamGrid = {
    'activation': ['relu'],
    'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
    'hidden_layer_sizes':
        []
        + [(25,), (50,), (75,), (100,), (125,), (150,), (175,), (200,), (225,), (250,), (275,), (300,)]
        + [(25, 5), (50, 10), (75, 15), (100, 20), (125, 25), (150, 30), (175, 35), (200, 40), (225, 45), (250, 50), (275, 55), (300, 60)]
        + [(25, 12), (50, 25), (75, 37), (100, 50), (125, 62), (150, 75), (175, 87), (200, 100), (225, 112), (250, 125), (275, 137), (300, 150)]
        + [(25, 25), (50, 50), (75, 75), (100, 100), (125, 125), (150, 150), (175, 175), (200, 200), (225, 225), (250, 250), (275, 275), (300, 300)]
}


def gen_thresholds(start: float, stop: float, step: float) -> List[float]:
    t = start
    thresholds = []
    while t <= stop:
        thresholds.append(t)
        t = t + step
    return thresholds


def grid_search_svm_rbf(x, y, k=5, scale=True, blind_x=None, blind_y=None, do_blind=False):
    print('Kernel', 'C', 'gamma', 'Min_Dec_Score', 'Max_Dec_Score', 'Threshold',
          'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC', 'BA')
    model = Model('SVM', x, y, k)
    for c in svmParamGridRBF['C']:
        for g in svmParamGridRBF['gamma']:
            param = {'kernel': 'rbf', 'C': c, 'gamma': g}
            model.learn(param, scale)
            min_des, max_des = model.get_decision_scores_k_fold(scale)
            thresholds = gen_thresholds(round(min_des, 1), round(max_des, 1), 0.1)

            acc = []
            sens = []
            spec = []
            f1s = []
            mcc = []
            ba = []
            for t in thresholds:
                result = model.predict_k_fold(scale=scale, threshold=t)
                acc.append(result[0])
                sens.append(result[1])
                spec.append(result[2])
                f1s.append(result[3])
                mcc.append(result[4])
                ba.append(result[5])
                # print(k, c, g, t, result[0], result[1], result[2], result[3], result[4], result[5])
            # diff = np.absolute(np.array(sens)-np.array(spec))
            # min_index = diff.tolist().index(diff.min())
            min_index = mcc.index(max(mcc))
            print('RBF', c, g, round(min_des, 1), round(max_des, 1), round(thresholds[min_index], 5),
                  round(acc[min_index], 3), round(sens[min_index], 3), round(spec[min_index], 3),
                  round(f1s[min_index], 3), round(mcc[min_index], 3), round(ba[min_index], 3))

            if do_blind:
                acc1, sens1, spec1, f1s1, mcc1, ba1 = model.predict_blind_data(blind_x, blind_y, scale=scale,
                                                                               threshold=thresholds[min_index])
                acc2, sens2, spec2, f1s2, mcc2, ba2 = model.predict_blind_without_cv(blind_x, blind_y, scale=scale,
                                                                                     threshold=thresholds[min_index])
                print('\t',
                      round(acc1, 3), round(sens1, 3), round(spec1, 3), round(f1s1, 3), round(mcc1, 3), round(ba1, 3),
                      round(acc2, 3), round(sens2, 3), round(spec2, 3), round(f1s2, 3), round(mcc2, 3), round(ba2, 3))


def grid_search_svm_rbf_wo_threshold(x, y, k=5, scale=True, blind_x=None, blind_y=None, do_blind=False):
    print('Kernel', 'C', 'gamma', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC', 'BA')
    model = Model('SVM', x, y, k)
    for c in svmParamGridRBF['C']:
        for g in svmParamGridRBF['gamma']:
            param = {'kernel': 'rbf', 'C': c, 'gamma': g}
            model.learn(param, scale)
            # min_des, max_des = model.get_decision_scores_k_fold(scale)
            # print('RBF', c, g, round(min_des, 3), round(max_des, 3))
            acc, sens, spec, f1s, mcc, ba = model.predict_k_fold(scale)
            print('RBF', c, g, round(acc, 3), round(sens, 3), round(spec, 3), round(f1s, 3), round(mcc, 3),
                  round(ba, 3))

            if do_blind:
                acc1, sens1, spec1, f1s1, mcc1, ba1 = model.predict_blind_data(blind_x, blind_y, scale=scale)
                acc2, sens2, spec2, f1s2, mcc2, ba2 = model.predict_blind_without_cv(blind_x, blind_y, scale=scale)
                print('\t', round(acc1, 3), round(sens1, 3), round(spec1, 3), round(f1s1, 3), round(mcc1, 3),
                      round(ba1, 3), round(acc2, 3), round(sens2, 3), round(spec2, 3), round(f1s2, 3), round(mcc2, 3),
                      round(ba2, 3))


def grid_search_svm_poly(x, y, k=5, scale=True, blind_x=None, blind_y=None, do_blind=False):
    print('Kernel', 'C', 'degree', 'gamma', 'coef0', 'Min_Dec_Score', 'Max_Dec_Score', 'Threshold',
          'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC', 'BA')
    model = Model('SVM', x, y, k)
    for c in svmParamGridPoly['C']:
        for deg in svmParamGridPoly['degree']:
            for g in svmParamGridPoly['gamma']:
                for coeff in svmParamGridPoly['coef0']:
                    param = {'kernel': 'poly', 'C': c, 'degree': deg, 'gamma': g, 'coef0': coeff}
                    model.learn(param, scale)
                    min_des, max_des = model.get_decision_scores_k_fold(scale)
                    thresholds = gen_thresholds(round(min_des, 1), round(max_des, 1), 0.1)

                    acc = []
                    sens = []
                    spec = []
                    f1s = []
                    mcc = []
                    ba = []
                    for t in thresholds:
                        result = model.predict_k_fold(scale=scale, threshold=t)
                        acc.append(result[0])
                        sens.append(result[1])
                        spec.append(result[2])
                        f1s.append(result[3])
                        mcc.append(result[4])
                        ba.append(result[5])
                        # print(k, c, g, t, result[0], result[1], result[2], result[3], result[4], result[5])
                    # diff = np.absolute(np.array(sens)-np.array(spec))
                    # min_index = diff.tolist().index(diff.min())
                    min_index = mcc.index(max(mcc))

                    print('poly', c, deg, g, coeff, round(min_des, 1), round(max_des, 1),
                          round(thresholds[min_index], 5), round(acc[min_index], 3), round(sens[min_index], 3),
                          round(spec[min_index], 3), round(f1s[min_index], 3), round(mcc[min_index], 3),
                          round(ba[min_index], 3))

                    if do_blind:
                        acc1, sens1, spec1, f1s1, mcc1, ba1 = model.predict_blind_data(blind_x, blind_y, scale=scale,
                                                                                       threshold=thresholds[min_index])
                        acc2, sens2, spec2, f1s2, mcc2, ba2 = model.predict_blind_without_cv(blind_x, blind_y,
                                                                                             scale=scale,
                                                                                             threshold=thresholds[
                                                                                                 min_index])
                        print('\t', round(acc1, 3), round(sens1, 3), round(spec1, 3), round(f1s1, 3), round(mcc1, 3),
                              round(ba1, 3), round(acc2, 3), round(sens2, 3), round(spec2, 3), round(f1s2, 3),
                              round(mcc2, 3), round(ba2, 3))


def grid_search_svm_poly_wo_threshold(x, y, k=5, scale=True, blind_x=None, blind_y=None, do_blind=False):
    print('Kernel', 'C', 'degree', 'gamma', 'coef0', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC', 'BA')
    model = Model('SVM', x, y, k)
    for c in svmParamGridPoly['C']:
        for deg in svmParamGridPoly['degree']:
            for g in svmParamGridPoly['gamma']:
                for coeff in svmParamGridPoly['coef0']:
                    param = {'kernel': 'poly', 'C': c, 'degree': deg, 'gamma': g, 'coef0': coeff}
                    model.learn(param, scale)
                    acc, sens, spec, f1s, mcc, ba = model.predict_k_fold(scale)
                    print('poly', c, deg, g, coeff, round(acc, 3), round(sens, 3), round(spec, 3), round(f1s, 3),
                          round(mcc, 3), round(ba, 3))

                    if do_blind:
                        acc1, sens1, spec1, f1s1, mcc1, ba1 = model.predict_blind_data(blind_x, blind_y, scale=scale)
                        acc2, sens2, spec2, f1s2, mcc2, ba2 = model.predict_blind_without_cv(blind_x, blind_y,
                                                                                             scale=scale)
                        print('\t', round(acc1, 3), round(sens1, 3), round(spec1, 3), round(f1s1, 3), round(mcc1, 3),
                              round(ba1, 3), round(acc2, 3), round(sens2, 3), round(spec2, 3), round(f1s2, 3),
                              round(mcc2, 3), round(ba2, 3))


def grid_search_svm_linear(x, y, k=5, scale=True, blind_x=None, blind_y=None, do_blind=False):
    print('Kernel', 'C', 'Min_Dec_Score', 'Max_Dec_Score', 'Threshold', 'Accuracy', 'Sensitivity', 'Specificity',
          'F1-score', 'MCC', 'BA')
    model = Model('SVM', x, y, k)
    for c in svmParamGridLinear['C']:
        param = {'kernel': 'linear', 'C': c}
        model.learn(param, scale)
        min_des, max_des = model.get_decision_scores_k_fold(scale)
        thresholds = gen_thresholds(round(min_des, 1), round(max_des, 1), 0.1)

        acc = []
        sens = []
        spec = []
        f1s = []
        mcc = []
        ba = []
        for t in thresholds:
            result = model.predict_k_fold(scale=scale, threshold=t)
            acc.append(result[0])
            sens.append(result[1])
            spec.append(result[2])
            f1s.append(result[3])
            mcc.append(result[4])
            ba.append(result[5])
            # print(k, c, g, t, result[0], result[1], result[2], result[3], result[4], result[5])
        # diff = np.absolute(np.array(sens)-np.array(spec))
        # min_index = diff.tolist().index(diff.min())
        min_index = mcc.index(max(mcc))

        print('linear', c, round(min_des, 1), round(max_des, 1), round(thresholds[min_index], 5),
              round(acc[min_index], 3), round(sens[min_index], 3), round(spec[min_index], 3), round(f1s[min_index], 3),
              round(mcc[min_index], 3), round(ba[min_index], 3))

        if do_blind:
            acc1, sens1, spec1, f1s1, mcc1, ba1 = model.predict_blind_data(blind_x, blind_y, scale=scale,
                                                                           threshold=thresholds[min_index])
            acc2, sens2, spec2, f1s2, mcc2, ba2 = model.predict_blind_without_cv(blind_x, blind_y, scale=scale,
                                                                                 threshold=thresholds[min_index])
            print('\t', round(acc1, 3), round(sens1, 3), round(spec1, 3), round(f1s1, 3), round(mcc1, 3), round(ba1, 3),
                  round(acc2, 3), round(sens2, 3), round(spec2, 3), round(f1s2, 3), round(mcc2, 3), round(ba2, 3))


def grid_search_svm_linear_wo_threshold(x, y, k=5, scale=True, blind_x=None, blind_y=None, do_blind=False):
    print('Kernel', 'C', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC', 'BA')
    model = Model('SVM', x, y, k)
    for c in svmParamGridLinear['C']:
        param = {'kernel': 'linear', 'C': c}
        model.learn(param, scale)
        acc, sens, spec, f1s, mcc, ba = model.predict_k_fold(scale)
        print('linear', c, round(acc, 3), round(sens, 3), round(spec, 3), round(f1s, 3), round(mcc, 3), round(ba, 3))

        if do_blind:
            acc1, sens1, spec1, f1s1, mcc1, ba1 = model.predict_blind_data(blind_x, blind_y, scale=scale)
            acc2, sens2, spec2, f1s2, mcc2, ba2 = model.predict_blind_without_cv(blind_x, blind_y, scale=scale)
            print('\t', round(acc1, 3), round(sens1, 3), round(spec1, 3), round(f1s1, 3), round(mcc1, 3), round(ba1, 3),
                  round(acc2, 3), round(sens2, 3), round(spec2, 3), round(f1s2, 3), round(mcc2, 3), round(ba2, 3))


def grid_search_rf_wo_threshold(x, y, k=5, scale=True, blind_x=None, blind_y=None, do_blind=False):
    print('n_estimators', 'max_depth', 'max_features',
          'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC', 'BA')
    model = Model('RF', x, y, k)
    for ne in rfParamGrid['n_estimators']:
        for md in rfParamGrid['max_depth']:
            for mf in rfParamGrid['max_features']:
                param = {'n_estimators': ne, 'max_depth': md, 'max_features': mf}
                model.learn(param, scale)
                acc, sens, spec, f1s, mcc, ba = model.predict_k_fold(scale)
                print(ne, md, mf, round(acc, 3), round(sens, 3), round(spec, 3), round(f1s, 3), round(mcc, 3),
                      round(ba, 3))

                if do_blind:
                    acc1, sens1, spec1, f1s1, mcc1, ba1 = model.predict_blind_data(blind_x, blind_y, scale=scale)
                    acc2, sens2, spec2, f1s2, mcc2, ba2 = model.predict_blind_without_cv(blind_x, blind_y, scale=scale)
                    print('\t', round(acc1, 3), round(sens1, 3), round(spec1, 3), round(f1s1, 3), round(mcc1, 3),
                          round(ba1, 3), round(acc2, 3), round(sens2, 3), round(spec2, 3), round(f1s2, 3),
                          round(mcc2, 3), round(ba2, 3))


def grid_search_mlp_wo_threshold(x, y, k=5, scale=True, blind_x=None, blind_y=None, do_blind=False):
    print('activation', 'layers', 'learning_rate', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC', 'BA')
    model = Model('MLP', x, y, k)
    for a in mlpParamGrid['activation']:
        for hls in mlpParamGrid['hidden_layer_sizes']:
            for lr in mlpParamGrid['learning_rate_init']:
                param = {'activation': a, 'hidden_layer_sizes': hls, 'learning_rate_init': lr}
                model.learn(param, scale)
                acc, sens, spec, f1s, mcc, ba = model.predict_k_fold(scale)
                print(a, hls, lr, round(acc, 3), round(sens, 3), round(spec, 3), round(f1s, 3), round(mcc, 3),
                      round(ba, 3))

                if do_blind:
                    acc1, sens1, spec1, f1s1, mcc1, ba1 = model.predict_blind_data(blind_x, blind_y, scale=scale)
                    acc2, sens2, spec2, f1s2, mcc2, ba2 = model.predict_blind_without_cv(blind_x, blind_y, scale=scale)
                    print('\t', round(acc1, 3), round(sens1, 3), round(spec1, 3), round(f1s1, 3), round(mcc1, 3),
                          round(ba1, 3), round(acc2, 3), round(sens2, 3), round(spec2, 3), round(f1s2, 3),
                          round(mcc2, 3), round(ba2, 3))


def get_optimal_model(feature_type: str, dataset: str, feature_extraction: str, estimator_id: str,
                      data: np.array, target: np.array, b_data: np.array = None, b_target: np.array = None,
                      verbose: bool = False) -> Model:
    from .optimal_params import optimal_params_radiological, optimal_thresholds_radiological
    from .optimal_params_combined import optimal_params_combined, optimal_thresholds_combined
    m = Model(estimator_id, data, target, k=5)
    if feature_type == 'radiological':
        m.learn(optimal_params_radiological[dataset][feature_extraction][estimator_id])
        threshold = optimal_thresholds_radiological[dataset][feature_extraction][estimator_id]
    elif feature_type == 'combined':
        m.learn(optimal_params_combined[dataset][feature_extraction][estimator_id])
        threshold = optimal_thresholds_combined[dataset][feature_extraction][estimator_id]
    if verbose:
        print(m.predict_k_fold(threshold=threshold))
        if b_data is not None and b_target is not None:
            print(m.predict_blind_without_cv(b_data, b_target, threshold=threshold))
        print()
    return m
