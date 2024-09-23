import pandas as pd
import matplotlib.pyplot as plt
import io
import warnings
from tqdm import tqdm

plt.rcParams['font.family'] = 'Noto Sans'
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

df = pd.read_csv('EPICorpus.csv')

label_inconsistency = df.groupby('text')['label'].nunique()
inconsistent_texts = label_inconsistency[label_inconsistency > 1].index
inconsistent_df = df[df['text'].isin(tqdm(inconsistent_texts, desc="Filtering for inconsistent texts"))]

demographics_columns = ['Language_annotator', 'Sex', 'Age', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality']
annotator_demographics = inconsistent_df[demographics_columns]

inconsistent_counts = inconsistent_df.groupby('text')['label'].nunique()

def sanitize_text(text):
    return text.replace('$', '')  # Remove dollar signs or modify as needed

inconsistent_counts.index = [sanitize_text(text) for text in inconsistent_counts.index]

inconsistent_counts.to_csv('inconsistent_counts.csv', index=False)
