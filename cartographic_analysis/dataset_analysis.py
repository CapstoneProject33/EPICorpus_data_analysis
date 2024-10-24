import csv
import pandas as pd


import pandas as pd

# Step 1: Read the CSV file
df = pd.read_csv('cartographic_analysis/output_agreement_levels.csv')

# Step 2: Get the number of unique text strings in the 'id' column
unique_ids = df['id'].nunique()

# Step 3: Display the count of unique IDs
print(f'The number of unique IDs is: {unique_ids}')
