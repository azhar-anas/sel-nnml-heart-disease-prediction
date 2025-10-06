import numpy as np
import optuna as opt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define objective functions and hyperparameter spaces for each model
def logistic_regression_objective(trial, x_train, y_train, cv_folds, metric_compare, n_jobs, random_state):
    solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'])
    C = trial.suggest_float('C', 1e-4, 10, log=True)
    model = LogisticRegression(solver=solver, C=C, max_iter=1000, random_state=random_state, n_jobs=n_jobs)
    return cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=metric_compare, n_jobs=n_jobs).mean()

def decision_tree_objective(trial, x_train, y_train, cv_folds, metric_compare, n_jobs, random_state):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    model = DecisionTreeClassifier(criterion=criterion, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state)
    return cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=metric_compare, n_jobs=n_jobs).mean()

def random_forest_objective(trial, x_train, y_train, cv_folds, metric_compare, n_jobs, random_state):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    model = RandomForestClassifier(criterion=criterion, max_features=max_features, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=n_jobs)
    return cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=metric_compare, n_jobs=n_jobs).mean()

def knn_objective(trial, x_train, y_train, cv_folds, metric_compare, n_jobs, random_state):
    algorithm = trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'brute'])
    n_neighbors = trial.suggest_int('n_neighbors', 3, 50)
    p = trial.suggest_int('p', 1, 2)
    model = KNeighborsClassifier(weights='uniform', algorithm=algorithm, n_neighbors=n_neighbors, p=p, n_jobs=n_jobs)
    return cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=metric_compare, n_jobs=n_jobs).mean()

def svm_objective(trial, x_train, y_train, cv_folds, metric_compare, n_jobs, random_state):
    kernel = trial.suggest_categorical('kernel', ['rbf', 'sigmoid', 'poly'])
    C = trial.suggest_float('C', 1e-4, 1e-2, log=True)
    degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
    model = SVC(kernel=kernel, gamma='scale', C=C, degree=degree, random_state=random_state)
    return cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=metric_compare, n_jobs=n_jobs).mean()

def adaboost_objective(trial, x_train, y_train, cv_folds, metric_compare, n_jobs, random_state):
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0, log=True)
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    return cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=metric_compare, n_jobs=n_jobs).mean()

def gradient_boosting_objective(trial, x_train, y_train, cv_folds, metric_compare, n_jobs, random_state):
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    model = GradientBoostingClassifier(criterion='friedman_mse', loss='log_loss', max_features=max_features, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, subsample=subsample, random_state=random_state)
    return cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=metric_compare, n_jobs=n_jobs).mean()

# Mapping model names to objective functions and model constructors
MODEL_OBJECTIVES = {
    'Logistic Regression': (logistic_regression_objective, LogisticRegression, {'max_iter': 1000}),
    'Decision Tree': (decision_tree_objective, DecisionTreeClassifier, {}),
    'Random Forest': (random_forest_objective, RandomForestClassifier, {}),
    'K-Nearest Neighbors': (knn_objective, KNeighborsClassifier, {'weights': 'uniform'}),
    'Support Vector Machine': (svm_objective, SVC, {'gamma': 'scale'}),
    'AdaBoost': (adaboost_objective, AdaBoostClassifier, {}),
    'Gradient Boosting': (gradient_boosting_objective, GradientBoostingClassifier, {'criterion': 'friedman_mse', 'loss': 'log_loss'}),
}

# Generalized tuning function
def base_model_tuning(model_name, x_train, y_train, sampler='TPESampler', cv_folds=5, iterations=100, metric_compare='accuracy', direction='maximize', n_jobs=-1, random_state=42):
    np.random.seed(random_state)

    # Error handling for unsupported models
    if model_name not in MODEL_OBJECTIVES:
        supported_models = list(MODEL_OBJECTIVES.keys())
        raise ValueError(f"Model '{model_name}' not supported. Supported models: {supported_models}")
    
    # Retrieve the objective function and model class
    objective_func, model_class, fixed_params = MODEL_OBJECTIVES[model_name]
    def objective(trial):
        return objective_func(trial, x_train, y_train, cv_folds, metric_compare, n_jobs, random_state)
    
    # Error handling for unsupported samplers
    supported_samplers = {
        'RandomSampler': opt.samplers.RandomSampler, # Random Search (RS) - Traditional Optimization
        'QMCSampler': opt.samplers.QMCSampler, # Quasi-Monte Carlo (QMC) - An upgrade of RS
        'CmaEsSampler': opt.samplers.CmaEsSampler, # Covariance Matrix Adaptation Evolution Strategy (CMA-ES) - Heuristic Optimization
        'GPSampler': opt.samplers.GPSampler, # Gaussian Process (GP) - Bayesian Optimization
        'TPESampler': opt.samplers.TPESampler, # Tree-structured Parzen Estimator (TPE) - Bayesian Optimization
    }
    if sampler not in supported_samplers:
        raise ValueError(f"Sampler '{sampler}' is not supported. Supported samplers: {list(supported_samplers.keys())}")
    
    # Initialize and run the Hyperparameter Optimization
    sampler_instance = supported_samplers[sampler](seed=random_state)
    study = opt.create_study(study_name=f'{model_name} Model Fine Tuning with {sampler}', direction=direction, sampler=sampler_instance)
    study.optimize(objective, n_trials=iterations, n_jobs=1, show_progress_bar=True)
    best_params = study.best_params

    # Prepare parameters for model instantiation
    model_params = {**best_params, **fixed_params}
    model_init_params = model_class().get_params().keys()
    if 'random_state' in model_init_params:
        model_params['random_state'] = random_state # add random_state to the model if applicable
    if 'n_jobs' in model_init_params:
        model_params['n_jobs'] = n_jobs # add n_jobs to the model if applicable

    # Return the best model configured with optimal hyperparameters
    model = model_class(**model_params)
    print(f'\nBest Hyperparameters for {model_name} Using {sampler}: {best_params}')
    print(f'Best {metric_compare}: {study.best_value:.4f}, at trial: {study.best_trial.number}')
    return model

def meta_model_tuning(models, x_train, y_train, x_test, y_test, sampler='TPESampler', iterations=100, metric_compare='accuracy',  direction='maximize', n_jobs=-1, random_state=42):
    np.random.seed(random_state)

    def objective(trial):
        # Suggest which base models to include
        selected_estimators = []
        for name, model in models.items():
            use_model = trial.suggest_categorical(f'use_{name}', [True, False])
            if use_model:
                selected_estimators.append((name, model))
        # At least 2 base models required for stacking
        if len(selected_estimators) < 2:
            return 0.0
        
        # Suggest hyperparameters for the meta-model (MLPClassifier)
        n_layers = trial.suggest_int('n_layers', 1, 5)
        neurons = [trial.suggest_int(f'n_neurons_{i}', 10, 100) for i in range(n_layers)]
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        learning_rate_init = trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True)
        alpha = trial.suggest_float('alpha', 0.0001, 0.01, log=True)
        meta_model = MLPClassifier(activation='relu', solver='adam', max_iter=300, hidden_layer_sizes=tuple(neurons), learning_rate=learning_rate, learning_rate_init=learning_rate_init, alpha=alpha, random_state=random_state)

        # Combine selected base models and meta-model into a StackingClassifier
        stacking_model = StackingClassifier(estimators=selected_estimators, final_estimator=meta_model, n_jobs=n_jobs)

        # Return the desired metric on the test set
        stacking_model.fit(x_train, y_train)
        y_pred = stacking_model.predict(x_test)
        if metric_compare == 'accuracy':
            return accuracy_score(y_test, y_pred)
        elif metric_compare == 'precision':
            return precision_score(y_test, y_pred, average='weighted', zero_division=0)
        elif metric_compare == 'recall':
            return recall_score(y_test, y_pred, average='weighted', zero_division=0)
        elif metric_compare == 'f1':
            return f1_score(y_test, y_pred, average='weighted', zero_division=0)
        else:
            raise ValueError(f"Metric '{metric_compare}' not supported in meta_model_tuning.")

    # Error handling for unsupported samplers
    supported_samplers = {
        'RandomSampler': opt.samplers.RandomSampler,
        'QMCSampler': opt.samplers.QMCSampler,
        'CmaEsSampler': opt.samplers.CmaEsSampler,
        'GPSampler': opt.samplers.GPSampler,
        'TPESampler': opt.samplers.TPESampler,
    }
    if sampler not in supported_samplers:
        raise ValueError(f"Sampler '{sampler}' is not supported. Supported samplers: {list(supported_samplers.keys())}")

    # Initialize and run the Hyperparameter Optimization for meta-model as stacking
    sampler_instance = supported_samplers[sampler](seed=random_state)
    study = opt.create_study(study_name=f'Meta Model Fine Tuning: Stacking with MLP ({sampler})', direction=direction, sampler=sampler_instance)
    study.optimize(objective, n_trials=iterations, n_jobs=1, show_progress_bar=True)
    best_params = study.best_params

    # Get selected estimators from best_params
    best_selected_estimators = []
    for name, model in models.items():
        if best_params.get(f'use_{name}', False):
            best_selected_estimators.append((name, model))

    # Get best meta-model parameters
    n_layers = best_params.pop('n_layers')
    hidden_layer_sizes = tuple(best_params.pop(f'n_neurons_{i}') for i in range(n_layers))
    mlp_params = {k: v for k, v in best_params.items() if not k.startswith('use_')}
    mlp_params.update({
        'activation': 'relu',
        'solver': 'adam',
        'hidden_layer_sizes': hidden_layer_sizes,
        'max_iter': 300,
        'random_state': random_state
    })
    best_meta_model = MLPClassifier(**mlp_params)

    # Final Stacking Classifier with best base models and meta-model
    best_stacking_model = StackingClassifier(
        estimators=best_selected_estimators,
        final_estimator=best_meta_model,
        n_jobs=n_jobs
    )

    print(f'\nSelected Base Models for Stacking using {sampler}:')
    for name, _ in best_selected_estimators:
        print(f'- {name}')
    print(f'Best Hyperparameters for Meta Model (MLP) using {sampler}: {mlp_params}')
    print(f'Best {metric_compare} on Test Set: {study.best_value:.4f}, at trial: {study.best_trial.number}')

    return best_stacking_model