import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def init_df(df,test_size=0.2,target_col='target',stratify=True,scaling=True,oversampling=True,undersampling=False,val=True,scaling_args={},smote_args={}):
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if stratify:
        strat = y
    else:
        strat = None

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    
    if val:
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=test_size,random_state=42)
    
    if scaling:
        
        scaler = StandardScaler(**scaling_args)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if val:
            X_val = scaler.transform(X_val)
    
    if oversampling:
       
        smote = SMOTE(random_state=42,**smote_args)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    elif undersampling:
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        
    if val:
        return X_train, X_test, X_val, y_val.values, y_train.values, y_test.values
    else:
        return X_train, X_test, y_train.values, y_test.values