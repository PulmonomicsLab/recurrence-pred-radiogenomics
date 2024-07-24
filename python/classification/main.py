import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from .grid_search import grid_search_svm_rbf, grid_search_svm_rbf_wo_threshold
from .grid_search import grid_search_svm_poly, grid_search_svm_poly_wo_threshold
from .grid_search import grid_search_svm_linear, grid_search_svm_linear_wo_threshold
from .grid_search import grid_search_rf_wo_threshold
from .grid_search import grid_search_mlp_wo_threshold
from .grid_search import get_optimal_model
from .roc import plot_roc_model_cv, plot_roc_model_blind

# non_augmented_range = 351, 468


cv_f_name = 'input/Intermediate/inception_intermediate_cv.tsv'
blind_f_name = 'input/Intermediate/inception_intermediate_blind.tsv'
# cv_f_name = 'input/Intermediate/xception_intermediate_cv.tsv'
# blind_f_name = 'input/Intermediate/xception_intermediate_blind.tsv'
print(cv_f_name + '\n' + blind_f_name)

cv_data = np.genfromtxt(cv_f_name, delimiter='\t', skip_header=1, usecols=range(1, 2049), dtype=np.float32)
cv_target = np.genfromtxt(cv_f_name, delimiter='\t', skip_header=1, usecols=range(0, 1), dtype=np.int64)
blind_data = np.genfromtxt(blind_f_name, delimiter='\t', skip_header=1, usecols=range(1, 2049), dtype=np.float32)
blind_target = np.genfromtxt(blind_f_name, delimiter='\t', skip_header=1, usecols=range(0, 1), dtype=np.int64)
print(cv_data.shape, cv_target.shape)
print(blind_data.shape, blind_target.shape)

os_total_data, os_total_target = SMOTE(sampling_strategy='auto', random_state=42).fit_resample(np.vstack((cv_data, blind_data)), np.hstack((cv_target, blind_target)))
os_cv_data = np.vstack((os_total_data[:cv_data.shape[0]], os_total_data[819:1120]))
os_cv_target = np.hstack((os_total_target[:cv_target.shape[0]], os_total_target[819: 1120]))
os_blind_data = np.vstack((os_total_data[cv_data.shape[0]:819], os_total_data[1120:]))
os_blind_target = np.hstack((os_total_target[cv_target.shape[0]:819], os_total_target[1120:]))
print(os_cv_data.shape, os_cv_target.shape)
print(os_blind_data.shape, os_blind_target.shape)

# print(np.array_equal(cv_data, os_total_data[:cv_data.shape[0]]))
# print(np.array_equal(cv_target, os_total_target[:cv_target.shape[0]]))
# print(np.array_equal(blind_data, os_total_data[cv_data.shape[0]:819]))
# print(np.array_equal(blind_target, os_total_target[cv_target.shape[0]:819]))
# print(np.sum(os_cv_target), np.sum(os_blind_target))

# grid_search_svm_rbf(cv_data, cv_target, blind_x=blind_data, blind_y=blind_target, do_blind=True)
# grid_search_svm_rbf_wo_threshold(cv_data, cv_target, blind_x=blind_data, blind_y=blind_target, do_blind=True)
# grid_search_svm_poly(cv_data, cv_target, blind_x=blind_data, blind_y=blind_target, do_blind=True)
# grid_search_svm_poly_wo_threshold(cv_data, cv_target, blind_x=blind_data, blind_y=blind_target, do_blind=True)
# grid_search_svm_linear(cv_data, cv_target, blind_x=blind_data, blind_y=blind_target, do_blind=True)
# grid_search_svm_linear_wo_threshold(cv_data, cv_target, blind_x=blind_data, blind_y=blind_target, do_blind=True)
# grid_search_rf_wo_threshold(cv_data, cv_target, blind_x=blind_data, blind_y=blind_target, do_blind=True)
# grid_search_mlp_wo_threshold(cv_data, cv_target, blind_x=blind_data, blind_y=blind_target, do_blind=True)

# grid_search_svm_rbf(os_cv_data, os_cv_target, blind_x=os_blind_data, blind_y=os_blind_target, do_blind=True)
# grid_search_svm_rbf_wo_threshold(os_cv_data, os_cv_target, blind_x=os_blind_data, blind_y=os_blind_target, do_blind=True)
# grid_search_svm_poly(os_cv_data, os_cv_target, blind_x=os_blind_data, blind_y=os_blind_target, do_blind=True)
# grid_search_svm_poly_wo_threshold(os_cv_data, os_cv_target, blind_x=os_blind_data, blind_y=os_blind_target, do_blind=True)
# grid_search_svm_linear(os_cv_data, os_cv_target, blind_x=os_blind_data, blind_y=os_blind_target, do_blind=True)
# grid_search_svm_linear_wo_threshold(os_cv_data, os_cv_target, blind_x=os_blind_data, blind_y=os_blind_target, do_blind=True)
# grid_search_rf_wo_threshold(os_cv_data, os_cv_target, blind_x=os_blind_data, blind_y=os_blind_target, do_blind=True)
# grid_search_mlp_wo_threshold(os_cv_data, os_cv_target, blind_x=os_blind_data, blind_y=os_blind_target, do_blind=True)

# m1 = get_optimal_model('radiological', 'original', 'inception', 'SVM', cv_data, cv_target, blind_data, blind_target, True)
# m2 = get_optimal_model('radiological', 'original', 'inception', 'RF', cv_data, cv_target, blind_data, blind_target, True)
# m3 = get_optimal_model('radiological', 'original', 'inception', 'MLP', cv_data, cv_target, blind_data, blind_target, True)
# m4 = get_optimal_model('radiological', 'original', 'xception', 'SVM', cv_data, cv_target, blind_data, blind_target, True)
# m5 = get_optimal_model('radiological', 'original', 'xception', 'RF', cv_data, cv_target, blind_data, blind_target, True)
# m6 = get_optimal_model('radiological', 'original', 'xception', 'MLP', cv_data, cv_target, blind_data, blind_target, True)
# m7 = get_optimal_model('radiological', 'smote', 'inception', 'SVM', os_cv_data, os_cv_target, os_blind_data, os_blind_target, True)
# m8 = get_optimal_model('radiological', 'smote', 'inception', 'RF', os_cv_data, os_cv_target, os_blind_data, os_blind_target, True)
# m9 = get_optimal_model('radiological', 'smote', 'inception', 'MLP', os_cv_data, os_cv_target, os_blind_data, os_blind_target, True)
# m10 = get_optimal_model('radiological', 'smote', 'xception', 'SVM', os_cv_data, os_cv_target, os_blind_data, os_blind_target, True)
# m11 = get_optimal_model('radiological', 'smote', 'xception', 'RF', os_cv_data, os_cv_target, os_blind_data, os_blind_target, True)
# m12 = get_optimal_model('radiological', 'smote', 'xception', 'MLP', os_cv_data, os_cv_target, os_blind_data, os_blind_target, True)

# plot_roc_model_cv(m1, verbose=True)
# plot_roc_model_blind(m1, blind_data, blind_target, verbose=True)
# plot_roc_model_cv(m2, verbose=True)
# plot_roc_model_blind(m2, blind_data, blind_target, verbose=True)
# plot_roc_model_cv(m3, verbose=True)
# plot_roc_model_blind(m3, blind_data, blind_target, verbose=True)
# plot_roc_model_cv(m4, verbose=True)
# plot_roc_model_blind(m4, blind_data, blind_target, verbose=True)
# plot_roc_model_cv(m5, verbose=True)
# plot_roc_model_blind(m5, blind_data, blind_target, verbose=True)
# plot_roc_model_cv(m6, verbose=True)
# plot_roc_model_blind(m6, blind_data, blind_target, verbose=True)
# plot_roc_model_cv(m7, verbose=True)
# plot_roc_model_blind(m7, os_blind_data, os_blind_target, verbose=True)
# plot_roc_model_cv(m8, verbose=True)
# plot_roc_model_blind(m8, os_blind_data, os_blind_target, verbose=True)
# plot_roc_model_cv(m9, verbose=True)
# plot_roc_model_blind(m9, os_blind_data, os_blind_target, verbose=True)
# plot_roc_model_cv(m10, verbose=True)
# plot_roc_model_blind(m10, os_blind_data, os_blind_target, verbose=True)
# plot_roc_model_cv(m11, verbose=True)
# plot_roc_model_blind(m11, os_blind_data, os_blind_target, verbose=True)
# plot_roc_model_cv(m12, verbose=True)
# plot_roc_model_blind(m12, os_blind_data, os_blind_target, verbose=True)
