import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load and merge datasets as before
annotation_df = pd.read_csv('annotation.csv')  # Replace with actual path
instance_df = pd.read_csv('instance.csv')      # Replace with actual path
merged_df = pd.merge(annotation_df, instance_df, on='instance_id')

# Step 3: Compute Cohen's kappa agreement level for each instance and assign levels
def compute_agreement_levels(df):
    # Group by 'instance_id' to get all labels for each instance
    grouped = df.groupby('instance_id')['label'].apply(list).reset_index(name='labels')

    # Function to calculate Cohen's kappa and determine agreement level
    def kappa_and_agreement(labels):
        label_counts = pd.Series(labels).value_counts()
        
        # Debugging: Check label distribution
        print("Label counts for current instance:", label_counts.to_dict())
        
        if len(label_counts) <= 1:  # Only one unique label
            return 1.0, 'Almost Perfect Agreement'  # All agree on the same label
        
        # Calculate Cohen's kappa using sklearn
        # Create a prediction array based on the majority
        majority_label = label_counts.idxmax()
        
        # Create predictions where majority is predicted for all, others are marked as 'other'
        preds = [majority_label if label == majority_label else 'other' for label in labels]
        
        # Generate true labels for kappa calculation
        true_labels = labels
        
        # Calculate Cohen's kappa
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

# Compute agreement levels
agreement_df = compute_agreement_levels(merged_df)

# Merge back the agreement levels with the original merged dataframe
final_df = pd.merge(merged_df, agreement_df[['instance_id', 'avg_kappa', 'agreement_level']], on='instance_id', how='left')

# Create the final DataFrame with the desired structure
output_df = pd.DataFrame({
    'id': final_df['instance_id'].astype(str) + "_test",
    'text': final_df['text'],
    'agreement_level': final_df['agreement_level'],
    'label': final_df['avg_kappa'].round(2),
    'text_id': final_df['instance_id']
})

# Step 7: Map agreement levels to numeric values
def convert_agreement_factor_to_num(value):
    if value == 'Almost Perfect Agreement':
        return 1.0
    elif value == 'Substantial Agreement':
        return 0.8
    elif value == 'Moderate Agreement':
        return 0.4  # Adjusted to allow 0.4 for moderate
    elif value in ['Fair Agreement', 'Slight Agreement', 'No Agreement']:
        return 0.2  # Adjusted to allow for lower factors
    else:
        return np.nan

# Create the agreement_factor based on the agreement_level
output_df['agreement_factor'] = output_df['agreement_level'].apply(convert_agreement_factor_to_num)

# Display the resulting dataframe with agreement levels and factors
print(output_df[['id', 'text', 'agreement_level', 'label', 'agreement_factor', 'text_id']])

# Write the output DataFrame to a CSV file
output_csv_path = 'output_agreement_levels.csv'
output_df.to_csv(output_csv_path, index=False)
print(f"Output written to {output_csv_path}")
