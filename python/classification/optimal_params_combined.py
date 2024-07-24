optimal_params_combined = {
    'original': {
        'inception': {
            'SVM': {'kernel': 'poly', 'C': 1, 'degree': 2, 'gamma': 0.001, 'coef0': 2},
            'RF': {'n_estimators': 600, 'max_depth': None, 'max_features': 0.25},
            'MLP': {'activation': 'relu', 'hidden_layer_sizes': (250,), 'learning_rate_init': 0.001}
        },
        'xception': {
            'SVM': {'kernel': 'poly', 'C': 1, 'degree': 2, 'gamma': 0.001, 'coef0': 3},
            'RF': {'n_estimators': 400, 'max_depth': None, 'max_features': 0.25},
            'MLP': {'activation': 'relu', 'hidden_layer_sizes': (275, 137), 'learning_rate_init': 0.001}
        }
    },
    'smote': {
        'inception': {
            'SVM': {'kernel': 'poly', 'C': 1, 'degree': 3, 'gamma': 0.001, 'coef0': 2},
            'RF': {'n_estimators': 200, 'max_depth': None, 'max_features': 0.5},
            'MLP': {'activation': 'relu', 'hidden_layer_sizes': (300, 60), 'learning_rate_init': 0.001}
        },
        'xception': {
            'SVM': {'kernel': 'poly', 'C': 1, 'degree': 2, 'gamma': 0.1, 'coef0': 3},
            'RF': {'n_estimators': 600, 'max_depth':  None, 'max_features': 0.25},
            'MLP': {'activation': 'relu', 'hidden_layer_sizes': (50,), 'learning_rate_init': 0.0001}
        }
    }
}

optimal_thresholds_combined = {
    'original': {
        'inception': {'SVM': -0.2, 'RF': None, 'MLP': None},
        'xception': {'SVM': -0.1, 'RF': None, 'MLP': None}
    },
    'smote': {
        'inception': {'SVM': 0.2, 'RF': None, 'MLP': None},
        'xception': {'SVM': 0, 'RF': None, 'MLP': None}
    }
}
