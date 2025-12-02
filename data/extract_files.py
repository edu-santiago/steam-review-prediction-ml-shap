import pandas as pd
    
final_reviews = pd.read_csv('https://drive.google.com/uc?export=download&id=1yd4Q4uDgoinFngpFLtekoQZHtudZGDta')
final_reviews.to_csv('final_reviews.csv')

df_tags = pd.read_csv('https://drive.google.com/uc?export=download&id=1RlxQ7gIWM86_nr_Z3lsS4TBBwVFmP4p_')
df_tags.to_csv('df_tags.csv')

true_user_data = pd.read_csv('https://drive.google.com/uc?export=download&id=1BlWzkcURHRHLxH-ilwP7jO2bp6-YQCNo')
true_user_data.to_csv('true_user_data.csv')