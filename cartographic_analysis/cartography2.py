import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load the single dataset
df = pd.read_csv('./csvs/EPICorpus.csv')  # Replace with actual path

# Step 3: Compute Cohen's kappa agreement level for each instance and assign levels
def compute_agreement_levels(df):
    # Group by 'id_original' to get all labels for each instance
    grouped = df.groupby('id_original')['label'].apply(list).reset_index(name='labels')

    # Function to calculate Cohen's kappa and determine agreement level
    def kappa_and_agreement(labels):
        label_counts = pd.Series(labels).value_counts()

        # Debugging: Check label distribution
        print("Label counts for current instance:", label_counts.to_dict())
        
        if len(label_counts) <= 1:  # Only one unique label
            return 1.0, 'Almost Perfect Agreement'  # All agree on the same label
        
        # Calculate Cohen's kappa using sklearn
        majority_label = label_counts.idxmax()
        preds = [majority_label if label == majority_label else 'other' for label in labels]
        true_labels = labels
        kappa = cohen_kappa_score(preds, true_labels)
        
        # Assign agreement level based on kappa score
        if kappa < 0:
            agreement_level = 'No Agreement'
        elif 0 <= kappa < 0.20:
            agreement_level = 'Slight Agreement'
        elif 0.20 <= kappa < 0.40:
            agreement_level = 'Fair Agreement'
        elif 0.40 <= kappa < 0.60:
            agreement_level = 'Moderate Agreement'
        elif 0.60 <= kappa < 0.80:
            agreement_level = 'Substantial Agreement'
        else:
            agreement_level = 'Almost Perfect Agreement'

        return kappa, agreement_level

    # Apply the kappa calculation for each instance
    grouped[['avg_kappa', 'agreement_level']] = grouped['labels'].apply(lambda x: pd.Series(kappa_and_agreement(x)))

    return grouped

# Compute agreement levels for each instance
agreement_df = compute_agreement_levels(df)

# Merge the agreement levels back into the original dataframe
final_df = pd.merge(df, agreement_df[['id_original', 'avg_kappa', 'agreement_level']], on='id_original', how='left')

# Create the final DataFrame with the desired structure, preserving demographic information
output_df = pd.DataFrame({
    'id': final_df['id_original'].astype(str) + "_test",
    'text': final_df['text'],
    'agreement_level': final_df['agreement_level'],
    'label': final_df['avg_kappa'].round(2),
    'text_id': final_df['id_original'],
    'user': final_df['user'],
    'timestamp': final_df['timestamp'],
    'source': final_df['source'],
    'subreddit': final_df['subreddit'],
    'parent_id_original': final_df['parent_id_original'],
    'parent_text': final_df['parent_text'],
    'Language_instance': final_df['Language_instance'],
    'Language_variety': final_df['Language_variety'],
    'Age': final_df['Age'],
    'Sex': final_df['Sex'],
    'Ethnicity simplified': final_df['Ethnicity simplified'],
    'Country of birth': final_df['Country of birth'],
    'Country of residence': final_df['Country of residence'],
    'Nationality': final_df['Nationality'],
    'Language_annotator': final_df['Language_annotator'],
    'Student status': final_df['Student status'],
    'Employment status': final_df['Employment status']
})

# Step 7: Map agreement levels to numeric values
def convert_agreement_factor_to_num(value):
    if value == 'Almost Perfect Agreement':
        return 1.0
    elif value == 'Substantial Agreement':
        return 0.8
    elif value == 'Moderate Agreement':
        return 0.4
    elif value in ['Fair Agreement', 'Slight Agreement', 'No Agreement']:
        return 0.2
    else:
        return np.nan

# Create the agreement_factor based on the agreement_level
output_df['agreement_factor'] = output_df['agreement_level'].apply(convert_agreement_factor_to_num)

# Remove duplicates based on 'id'
output_df = output_df.drop_duplicates(subset='id', keep='first')

# Display the resulting dataframe with agreement levels and factors
print(output_df[['id', 'text', 'agreement_level', 'label', 'agreement_factor', 'text_id', 'user', 'Age', 'Sex', 'Ethnicity simplified']])

# Write the output DataFrame to a CSV file
output_csv_path = './cartographic_analysis/output_agreement_levels2.csv'
output_df.to_csv(output_csv_path, index=False)
print(f"Output written to {output_csv_path}")
