from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.neighbors import KNeighborsClassifier
from statistics import mean
import csv
import copy


class Model:
    
    def __init__(self, e_id, x, y, k=5, do_shuffle=False):
        self.estimators = []
        self.total_estimator = None
        self.train_indices = []
        self.test_indices = []
        self.estimator_id = e_id
        if do_shuffle:
            self.data = shuffle(x, random_state=42)
            self.target = shuffle(y, random_state=42)
        else:
            self.data = x
            self.target = y
        self.n_folds = k
        self.split_cv_folds()
        self.dataScaler = StandardScaler().fit(self.data)
        # self.targetScaler = StandardScaler().fit(self.target.reshape(-1,1))

    def get_decision_function(self, estimator):
        if self.estimator_id == 'SVM':
            return estimator.decision_function
        elif self.estimator_id == 'RF':
            return estimator.predict_proba
        elif self.estimator_id == 'MLP':
            return estimator.predict_proba
        else:
            raise ValueError('Error !!! Invalid estimator ID. Must be {SVM, RF, MLP}')

    def create_estimator(self, params):
        estimator = None
        if self.estimator_id == 'SVM':  # SVM
            estimator = SVC(class_weight='balanced')
        elif self.estimator_id == 'RF':  # RF
            estimator = RF(class_weight='balanced_subsample', random_state=42, n_jobs=-1)
        elif self.estimator_id == 'MLP':  # MLP
            estimator = MLP(solver='adam', max_iter=10000, n_iter_no_change=10, random_state=42)
        else:
            raise ValueError('Error !!! Invalid estimator ID. Must be {SVM, RF, MLP}')
        estimator.set_params(**params)
        return estimator

    def split_cv_folds(self):
        if self.n_folds == 1:
            self.train_indices = [range(self.data.shape[0])]
            self.test_indices = [range(self.data.shape[0])]
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            for train_index, test_index in skf.split(self.data, self.target):
                self.train_indices.append(train_index)
                self.test_indices.append(test_index)

    def learn_without_cv(self, params, scale=True):
        estimator = self.create_estimator(params)
        if scale:
            estimator.fit(self.dataScaler.transform(self.data), self.target)
        else:
            estimator.fit(self.data, self.target)
        self.total_estimator = copy.deepcopy(estimator)

    def learn_k_fold(self, params, scale=True):
        self.estimators = []
        for f in range(self.n_folds):
            estimator = self.create_estimator(params)
            if scale:
                estimator.fit(
                    self.dataScaler.transform(self.data[self.train_indices[f]]),
                    self.target[self.train_indices[f]]
                )
            else:
                estimator.fit(self.data[self.train_indices[f]], self.target[self.train_indices[f]])
            self.estimators.append(copy.deepcopy(estimator))
    
    def learn(self, params, scale=True):
        self.learn_without_cv(params, scale)
        self.learn_k_fold(params, scale)
    
    def predict_one_fold(self, estimator, X_test, scale=True, threshold=None):
        if scale:
            X_test = self.dataScaler.transform(X_test)
            # y_pred = estimator.predict(self.dataScaler.transform(X_test))

        # df = estimator.decision_function(X_test)
        
        if threshold is None:
            y_pred = estimator.predict(X_test)
        else:
            if self.estimator_id == 'RF':
                des = estimator.predict_proba(X_test)
                y_pred = []
                for val in des:
                    if val[1] > threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            if self.estimator_id == 'GNB':
                des = estimator.predict_proba(X_test)
                y_pred = []
                for val in des:
                    if val[1] > threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            if self.estimator_id == 'KNN':
                des = estimator.predict_proba(X_test)
                y_pred = []
                for val in des:
                    if val[1] > threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            if self.estimator_id == 'MLP':
                des = estimator.predict_proba(X_test)
                y_pred = []
                for val in des:
                    if val[1] > threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            elif self.estimator_id == 'SVM':
                des = estimator.decision_function(X_test)
                y_pred = []
                for val in des:
                    if val > threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
        
        # print(y_pred.shape)
        return y_pred
    
    def predict_k_fold(self, scale=True, threshold=None):
        sensitivities = []
        specificities = []
        accuracies = []
        f1_scores = []
        mccs = []
        balanced_accuracies = []
        for f in range(self.n_folds):
            y_test = self.target[self.test_indices[f]]
            y_pred = self.predict_one_fold(
                self.estimators[f],
                self.data[self.test_indices[f]],
                scale=scale, threshold=threshold
            )
            sensitivities.append(recall_score(y_test, y_pred))
            accuracies.append(accuracy_score(y_test, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificities.append((tn/1.0) / (tn+fp))
            f1_scores.append(f1_score(y_test, y_pred))
            mccs.append(matthews_corrcoef(y_test, y_pred))
            balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        # print((mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs), mean(balanced_accuracies)))
        return (mean(accuracies), mean(sensitivities), mean(specificities),
                mean(f1_scores), mean(mccs), mean(balanced_accuracies))
    
    def predict_blind_data(self, b_data, b_target, scale=True, threshold=None):
        sensitivities = []
        specificities = []
        accuracies = []
        f1_scores = []
        mccs = []
        balanced_accuracies = []
        for f in range(self.n_folds):
            y_pred = self.predict_one_fold(self.estimators[f], b_data, scale=scale, threshold=threshold)
            sensitivities.append(recall_score(b_target, y_pred))
            accuracies.append(accuracy_score(b_target, y_pred))
            tn, fp, fn, tp = confusion_matrix(b_target, y_pred).ravel()
            specificities.append((tn/1.0) / (tn+fp))
            f1_scores.append(f1_score(b_target, y_pred))
            mccs.append(matthews_corrcoef(b_target, y_pred))
            balanced_accuracies.append(balanced_accuracy_score(b_target, y_pred))
        # print((mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs), mean(balanced_accuracies)))
        return (mean(accuracies), mean(sensitivities), mean(specificities),
                mean(f1_scores), mean(mccs), mean(balanced_accuracies))

    def predict_blind_without_cv(self, b_data, b_target, scale=True, threshold=None):
        y_pred = self.predict_one_fold(self.total_estimator, b_data, scale=scale, threshold=threshold)
        # y_pred = self.total_estimator.predict(b_data)
        tn, fp, fn, tp = confusion_matrix(b_target, y_pred).ravel()
        # specificities.append((tn/1.0) / (tn+fp))
        return accuracy_score(b_target, y_pred), recall_score(b_target, y_pred), ((tn/1.0) / (tn+fp)), \
            f1_score(b_target, y_pred), matthews_corrcoef(b_target, y_pred), balanced_accuracy_score(b_target, y_pred)
    
    def get_decision_scores_k_fold(self, scale=True):
        min_des = float('inf')
        max_des = float('-inf')
        for f in range(self.n_folds):
            if scale:
                X_test = self.dataScaler.transform(self.data[self.test_indices[f]])
            else:
                X_test = self.data[self.test_indices[f]]
            
            des = self.estimators[f].decision_function(X_test)
            if min(des) < min_des:
                min_des = min(des)
            if max(des) > max_des:
                max_des = max(des)
        
        return min_des, max_des

    def get_decision_score(self, estimator, x_test, scale=True):
        decision_function = self.get_decision_function(estimator)
        if scale:
            x_test = self.dataScaler.transform(x_test)
        else:
            x_test = self.data[x_test]
        des = decision_function(x_test)
        return des
