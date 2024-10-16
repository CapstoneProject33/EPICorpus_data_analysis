import os
import pandas as pd
from trash.cartography_utils import (
    create_annotations_per_annotator_plot,
    create_label_histogram_plot,
    create_label_distribution_heatmap,
    create_correlation_matrix_plot,
    create_outlier_detection_plot,
    create_pie_charts_for_demographics,
    collect_dataset_statistics
)
from trash.readme_writer import generate_readme

# Load the dataset
df = pd.read_csv('EPICorpus.csv')

# Directory for storing images
if not os.path.exists('images'):
    os.makedirs('images')

# Collect dataset statistics
stats = collect_dataset_statistics(df)

# Generate plots
annotations_per_annotator_image = create_annotations_per_annotator_plot(df)
label_histogram_image = create_label_histogram_plot(df)
annotator_label_heatmap_image = create_label_distribution_heatmap(df)
correlation_matrix_image = create_correlation_matrix_plot(df)
outlier_boxplot_image = create_outlier_detection_plot(df)
demographic_pie_charts = create_pie_charts_for_demographics(df)

# Write everything to README.md
generate_readme(
    stats,
    annotations_per_annotator_image,
    label_histogram_image,
    annotator_label_heatmap_image,
    correlation_matrix_image,
    outlier_boxplot_image,
    demographic_pie_charts
)
