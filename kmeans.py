import os
import argparse
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def process_file(filepath, output_dir):
    # Read the Excel file
    df = pd.read_excel(filepath)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=10, random_state=0).fit(df[['Weighted_Sentiment_Score']])
    df['Cluster'] = kmeans.labels_

    # Save the modified DataFrame to a new Excel file
    filename = os.path.basename(filepath)
    new_filename = os.path.splitext(filename)[0] + '_k.xlsx'
    df.to_excel(os.path.join(output_dir, new_filename), index=False)

    # Generate and save the dendrogram
    Z = linkage(df[['Weighted_Sentiment_Score']], 'ward')
    plt.figure()
    dendrogram(Z)
    dendrogram_filename = os.path.splitext(filename)[0] + '_dendrogram.png'
    plt.savefig(os.path.join(output_dir, dendrogram_filename))
    plt.close()

def main(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(input_dir, filename)
            process_file(filepath, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='K-means clustering and dendrogram generation for sentiment scores in Excel files.')
    parser.add_argument('--inputdir', required=True, help='Path to the input directory containing Excel files.')
    parser.add_argument('--outputdir', required=True, help='Path to the output directory to save the results.')
    args = parser.parse_args()
    
    main(args.inputdir, args.outputdir)
