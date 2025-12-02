import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score,make_scorer
from imblearn.pipeline import Pipeline

from utils.utils import *
from utils.pre_processing import *

sklearn_scores = {'accuracy_score':accuracy_score,
                  'precision_score':precision_score,
                  'recall_score':recall_score,
                  'f1_score':f1_score}

weight_variable = {
    'LogisticRegression':'class_weight',
    'RandomForest':'class_weight',
    'XGBoost':'scale_pos_weight',
    'LightGBM':'scale_pos_weight'
}

def load_full_dataset(oversampling=True,undersampling=False):
    
    df = load_data('data/df_tags','data/final_reviews')

    X_train, X_test, X_val, y_val, y_train, y_test = init_df(df,test_size=0.2,target_col='target',stratify=True,scaling=True,oversampling=oversampling,undersampling=undersampling)
    X_train, X_test, X_val, to_remove = remove_highly_correlated_features(X_train, X_test, X_val, threshold=0.9)
    df = df.drop(columns=df.columns[to_remove])
    
    scale_pos_weight = (1 - y_train).sum() / y_train.sum() if not (oversampling or undersampling) else 1

    return X_train, X_test, X_val, y_val, y_train, y_test, to_remove, df, scale_pos_weight

def create_dataset(oversampling=True,undersampling=True,no_sampling=True):

    datasets = {}

    if oversampling:

        (X_train, X_test, X_val, y_val, y_train, y_test, to_remove, df, scale_pos_weight) = load_full_dataset(oversampling=True,undersampling=False)
        
        datasets['oversampling'] = {
            'X_train':X_train,
            'X_test':X_test,
            'X_val':X_val,
            'y_val':y_val,
            'y_train':y_train,
            'y_test':y_test,
            'to_remove':to_remove,
            'df':df,
            'scale_pos_weight':scale_pos_weight
                                  }

    if undersampling:
    
        (X_train, X_test, X_val, y_val, y_train, y_test, to_remove, df, scale_pos_weight) = load_full_dataset(oversampling=False,undersampling=True)
        
        datasets['undersampling'] = {
            'X_train':X_train,
            'X_test':X_test,
            'X_val':X_val,
            'y_val':y_val,
            'y_train':y_train,
            'y_test':y_test,
            'to_remove':to_remove,
            'df':df,
            'scale_pos_weight':scale_pos_weight
                                  }

    if no_sampling:
    
        (X_train, X_test, X_val, y_val, y_train, y_test, to_remove, df, scale_pos_weight) = load_full_dataset(oversampling=False,undersampling=False)
        
        datasets['no_sampling'] = {
            'X_train':X_train,
            'X_test':X_test,
            'X_val':X_val,
            'y_val':y_val,
            'y_train':y_train,
            'y_test':y_test,
            'to_remove':to_remove,
            'df':df,
            'scale_pos_weight':scale_pos_weight
                                  }

    return datasets

def run_ml_models(models,x_train,y_train,x_val,y_val,folds=5,scorer='f1_score',n_jobs=-1):

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    scoring = make_scorer(sklearn_scores.get(scorer))

    best_models = {}
    
    for model_name, (model, params) in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(x_train, y_train)
        
        
        print(f"{model_name}: Best F1 Score = {grid_search.best_score_}, Best Params = {grid_search.best_params_}")

        f1 = f1_score(y_val, grid_search.best_estimator_.predict(x_val))
        print(f"{model_name} on validation set: F1 Score = {f1}")

        best_models[model_name] = {'model':grid_search.best_estimator_,
                                   'params':grid_search.best_params_,
                                   'test_score':grid_search.best_score_,
                                   'val_score':f1}

    return best_models