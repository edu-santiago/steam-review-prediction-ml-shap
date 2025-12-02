import pandas as pd
import numpy as np


def corr_filter(df, threshold=0.7, target='target'):
    correlation_matrix = df.corr().abs()
    upper_triangle_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    high_correlation_pairs = [(column, row) for column in upper_triangle_matrix.columns for row in upper_triangle_matrix.index if upper_triangle_matrix.at[row, column] > threshold]
    
    variances = df.var()
    to_drop = set()
    for feature1, feature2 in high_correlation_pairs:
        if variances[feature1] < variances[feature2]:
            to_drop.add(feature1)
        else:
            to_drop.add(feature2)
    
    df_reduced = df.drop(columns=to_drop)
    
    return df_reduced


def remove_highly_correlated_features(X_train, X_test, X_val=None, val=True, threshold=0.9):
    # Step 1: Calculate the correlation matrix
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    
    # Step 2: Identify pairs of features with absolute correlation higher than the threshold
    to_remove = set()
    num_features = corr_matrix.shape[0]
    
    for i in range(num_features):
        for j in range(i+1, num_features):
            if abs(corr_matrix[i, j]) > threshold:
                # Step 3: Compare variance and mark the one with smaller variance for removal
                if np.var(X_train[:, i]) < np.var(X_train[:, j]):
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    
    # Convert to list and sort to ensure consistent order
    to_remove = sorted(list(to_remove))
    
    # Step 4: Remove columns from X_train, X_test, and X_val
    X_train_reduced = np.delete(X_train, to_remove, axis=1)
    X_test_reduced = np.delete(X_test, to_remove, axis=1)
    if val:
        X_val_reduced = np.delete(X_val, to_remove, axis=1)
    
        return X_train_reduced, X_test_reduced, X_val_reduced, to_remove
    else:
        return X_train_reduced, X_test_reduced, to_remove

def load_data(main_df='df_tags',reviews_df='final_reviews'):

    main_df_csv = main_df + '.csv'
    reviews_df_csv = reviews_df + '.csv'
    
    final_df = pd.read_csv(main_df_csv)
    final_df = final_df[[col for col in final_df.columns if 'Unnamed' not in col]]
    
    reviews = pd.read_csv(reviews_df_csv)
    reviews = reviews[[col for col in reviews.columns if 'Unnamed' not in col]].rename(columns={'steamid':'id','voted_up':'target'})
    reviews = reviews[['id','num_reviews','playtime_at_review','target']]
    reviews['target'] = 1 - reviews['target']
    
    df = final_df.merge(reviews,on='id',how='inner').drop(columns='id')

    return df