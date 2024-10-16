import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Load the dataset
df = pd.read_csv('EPICorpus.csv')

# Directory for storing images
if not os.path.exists('images'):
    os.makedirs('images')

# 1. Number of annotators
num_annotators = df['Language_annotator'].nunique()

# 2. Number of annotations
num_annotations = len(df)

# 3. Histogram of number of annotations per annotator
annotations_per_annotator = df.groupby('Language_annotator').size()
plt.figure(figsize=(10, 6))
annotations_per_annotator.plot(kind='bar', color='skyblue')
plt.title('Histogram: Number of Annotations per Annotator')
plt.xlabel('Annotator')
plt.ylabel('Number of Annotations')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the figure as an image
annotations_per_annotator_image = 'images/annotations_per_annotator.png'
plt.savefig(annotations_per_annotator_image)
plt.close()

# 4. How many distinct labels are there in the dataset?
distinct_labels = df['label'].nunique()

# Determine if it's binary, multiclass, or continuous classification
classification_type = "binary" if distinct_labels == 2 else ("continuous" if df['label'].dtype in ['float64', 'int64'] else "multi-class")

# 5. Histogram of label values
plt.figure(figsize=(10, 6))
df['label'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Histogram: Label Values')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the figure as an image
label_histogram_image = 'images/label_histogram.png'
plt.savefig(label_histogram_image)
plt.close()

# 6. Demographic features counts (if available)
demographic_columns = ['Sex', 'Age', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality']

demographic_markdown = ""
for column in demographic_columns:
    if column in df.columns:
        demographic_markdown += f"\n### Demographic feature: {column}\n"
        demographic_markdown += df[column].value_counts().to_markdown() + "\n\n"

# Write everything to README.md
with open("README.md", "w") as readme_file:
    readme_file.write("# EPIC Dataset Analysis\n")
    readme_file.write(f"### Number of Unique Annotators: {num_annotators}\n")
    readme_file.write(f"### Total Number of Annotations: {num_annotations}\n")
    readme_file.write(f"### Number of Distinct Labels: {distinct_labels}\n")
    readme_file.write(f"### Classification Type: {classification_type}\n")

    readme_file.write("\n## Visualizations\n")
    
    # Insert image for annotations per annotator
    readme_file.write(f"![Annotations per Annotator]({annotations_per_annotator_image})\n\n")
    
    # Insert image for label histogram
    readme_file.write(f"![Label Histogram]({label_histogram_image})\n\n")

    # Append demographic features
    if demographic_markdown:
        readme_file.write("\n## Annotator Demographics\n")
        readme_file.write(demographic_markdown)

