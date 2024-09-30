import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Function to create histogram of annotations per annotator
def create_annotations_per_annotator_plot(df):
    annotations_per_annotator = df.groupby('Language_annotator').size()
    plt.figure(figsize=(10, 6))
    annotations_per_annotator.plot(kind='bar', color='skyblue')
    plt.title('Histogram: Number of Annotations per Annotator')
    plt.xlabel('Annotator')
    plt.ylabel('Number of Annotations')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the figure as an image
    image_path = 'images/annotations_per_annotator.png'
    plt.savefig(image_path)
    plt.close()

    return image_path

# 2. Function to create histogram of label values
def create_label_histogram_plot(df):
    plt.figure(figsize=(10, 6))
    df['label'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Histogram: Label Values')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the figure as an image
    image_path = 'images/label_histogram.png'
    plt.savefig(image_path)
    plt.close()

    return image_path

# 3. Function to create a heatmap of label distribution per annotator
def create_label_distribution_heatmap(df):
    annotator_label_distribution = pd.crosstab(df['Language_annotator'], df['label'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(annotator_label_distribution, annot=True, cmap="YlGnBu", fmt="d")
    plt.title('Heatmap: Label Distribution per Annotator')
    plt.xlabel('Label')
    plt.ylabel('Annotator')
    plt.tight_layout()

    # Save the heatmap
    image_path = 'images/annotator_label_heatmap.png'
    plt.savefig(image_path)
    plt.close()

    return image_path

# 4. Function to create correlation matrix for numerical features
def create_correlation_matrix_plot(df):
    numerical_columns = ['Age']  # Add other numeric columns if needed
    if 'label' in df.columns and pd.api.types.is_numeric_dtype(df['label']):
        numerical_columns.append('label')

    # Select only numerical columns, safely handling non-numeric data
    df_numerical = df[numerical_columns].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, non-convertible become NaN

    if not df_numerical.empty and df_numerical.isnull().sum().sum() < len(df_numerical):
        plt.figure(figsize=(10, 8))
        correlation_matrix = df_numerical.corr()  # Compute correlation matrix
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix: Numerical Features')
        plt.tight_layout()

        # Save the correlation matrix
        image_path = 'images/correlation_matrix.png'
        plt.savefig(image_path)
        plt.close()

        return image_path
    else:
        print("No numerical data available for correlation.")
        return None

# 5. Function to detect outliers using a boxplot (e.g., for Age)
def create_outlier_detection_plot(df):
    if 'Age' in df.columns and pd.api.types.is_numeric_dtype(df['Age']):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['Age'])
        plt.title('Boxplot: Age (Outlier Detection)')
        plt.tight_layout()

        # Save the boxplot
        image_path = 'images/age_outliers.png'
        plt.savefig(image_path)
        plt.close()

        return image_path
    return None

# 6. Function to create pie charts for demographic features
def create_pie_charts_for_demographics(df):
    demographic_columns = ['Sex', 'Age', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality']
    pie_chart_images = {}

    for column in demographic_columns:
        if column in df.columns and df[column].nunique() <= 10:  # Only plot for categories with fewer than 10 unique values
            plt.figure(figsize=(7, 7))
            df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
            plt.title(f'Pie Chart: {column}')
            plt.ylabel('')
            plt.tight_layout()

            # Save the pie chart
            image_path = f'images/{column}_pie_chart.png'
            plt.savefig(image_path)
            plt.close()

            pie_chart_images[column] = image_path

    return pie_chart_images

# Function to collect dataset statistics
def collect_dataset_statistics(df):
    num_annotators = df['Language_annotator'].nunique()
    num_annotations = len(df)
    distinct_labels = df['label'].nunique()

    classification_type = "binary" if distinct_labels == 2 else ("continuous" if df['label'].dtype in ['float64', 'int64'] else "multi-class")

    return {
        'num_annotators': num_annotators,
        'num_annotations': num_annotations,
        'distinct_labels': distinct_labels,
        'classification_type': classification_type
    }
