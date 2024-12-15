import pandas as pd
import re

def normalize_title(title):
    return re.sub(r'^[^a-zA-Z0-9]+', '', title).strip()

df = pd.read_csv('data/v1.csv')

df['normalized_title'] = df['title'].apply(normalize_title)

df = df.drop_duplicates(subset='normalized_title')

df = df.drop(columns=['normalized_title'])

df.to_csv('v1_sem_duplicatas.csv', index=False)