import os

# Function to generate README
def generate_readme(stats, annotations_per_annotator_image, label_histogram_image, annotator_label_heatmap_image, correlation_matrix_image, outlier_boxplot_image, demographic_pie_charts):
    with open("README.md", "w") as readme_file:
        readme_file.write("# EPIC Dataset Analysis\n")
        readme_file.write(f"### Number of Unique Annotators: {stats['num_annotators']}\n")
        readme_file.write(f"### Total Number of Annotations: {stats['num_annotations']}\n")
        readme_file.write(f"### Number of Distinct Labels: {stats['distinct_labels']}\n")
        readme_file.write(f"### Classification Type: {stats['classification_type']}\n")

        readme_file.write("\n## Visualizations\n")
        
        # Insert image for annotations per annotator
        if annotations_per_annotator_image:
            readme_file.write(f"![Annotations per Annotator]({annotations_per_annotator_image})\n\n")
        
        # Insert image for label histogram
        if label_histogram_image:
            readme_file.write(f"![Label Histogram]({label_histogram_image})\n\n")
        
        # Insert heatmap for label distribution per annotator
        if annotator_label_heatmap_image:
            readme_file.write(f"![Label Distribution Heatmap]({annotator_label_heatmap_image})\n\n")

        # Insert correlation matrix
        if correlation_matrix_image:
            readme_file.write(f"![Correlation Matrix]({correlation_matrix_image})\n\n")

        # Insert age outlier boxplot if applicable
        if outlier_boxplot_image:
            readme_file.write(f"![Age Outlier Boxplot]({outlier_boxplot_image})\n\n")

        # Insert demographic feature distributions (Pie Charts)
        if demographic_pie_charts:
            for column, pie_chart_image in demographic_pie_charts.items():
                if os.path.exists(pie_chart_image):
                    readme_file.write(f"![Pie Chart for {column}]({pie_chart_image})\n\n")
