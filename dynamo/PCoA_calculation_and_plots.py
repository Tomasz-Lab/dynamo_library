import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from skbio.stats.composition import clr


class PCoACalculationAndPlots:

    @staticmethod
    def create_normalized_aitchinson_distance_matrix(clr_df: np.ndarray) -> pd.DataFrame:
        if not isinstance(clr_df, np.ndarray):
            raise ValueError("Input 'clr_df' must be a numpy np.ndarray.")

        X1_idx = []
        X2_idx = []
        norm_aitchison_distance = []

        for i in range(len(clr_df)):
            for j in range(len(clr_df)):
                x1 = clr_df[i]
                x2 = clr_df[j]

                dist = np.linalg.norm(x1 - x2)
                dist = 0.5 * (np.std(x1 - x2) ** 2) / (np.std(x1) ** 2 + np.std(x2) ** 2)

                X1_idx.append(i)
                X2_idx.append(j)
                norm_aitchison_distance.append(dist)

        norm_aitchison_distance_df = pd.DataFrame(
            list(zip(X1_idx, X2_idx, norm_aitchison_distance)),
            columns=['x1', 'x2', 'normalized_aitchinson_distance']
        )

        norm_aitchison_distance_matrix = norm_aitchison_distance_df.pivot(
            index='x1',
            columns='x2',
            values='normalized_aitchinson_distance'
        )

        return norm_aitchison_distance_matrix

    @staticmethod
    def run_pca(distance_matrix: pd.DataFrame, n_components: int = 2, subject_column: np.ndarray = None) -> tuple:
        if not isinstance(distance_matrix, pd.DataFrame):
            raise ValueError("Input 'distance_matrix' must be a pandas DataFrame.")

        if distance_matrix.empty:
            raise ValueError("Input 'distance_matrix' DataFrame is empty.")

        if n_components < 1 or n_components > distance_matrix.shape[1]:
            raise ValueError(
                "'n_components' must be a positive integer less than or equal to the number of columns in 'distance_matrix'.")

        pca = PCA(n_components=n_components)
        pca.fit(distance_matrix.values)
        explained_variance_ratio = pca.explained_variance_ratio_
        pca_results = pd.DataFrame(pca.transform(distance_matrix.values))
        pca_results.columns = [f'PC{i + 1}' for i in range(n_components)]
        pca_results['timepoint'] = pca_results.index.astype(float)

        if subject_column is not None:
            pca_results['subject'] = subject_column

        return pca_results, explained_variance_ratio

    @staticmethod
    def visualize_pca(pca_results: pd.DataFrame, explained_variance_ratio: np.ndarray, individual: bool = True, base_color: str = 'blue') -> None:
        if not isinstance(pca_results, pd.DataFrame):
            raise ValueError("Input 'pca_results' must be a pandas DataFrame.")

        if not isinstance(explained_variance_ratio, np.ndarray):
            raise ValueError("'explained_variance_ratio' must be a numpy ndarray.")

        plt.figure(figsize=(10, 5))
        handles, labels = plt.gca().get_legend_handles_labels()

        if individual:
            sns.scatterplot(data=pca_results, x='PC1', y='PC2', hue='timepoint', palette=base_color, s=80, edgecolor='k',
                            lw=1.4, alpha=1)
            plt.title('Individual PCA Results', fontsize=16)
            plt.legend(handles=handles, labels=labels, title='Time point', fontsize=12)
        else:
            sns.scatterplot(data=pca_results, x='PC1', y='PC2', hue='subject', palette='husl', s=80, edgecolor='k',
                            lw=1.4, alpha=1)
            plt.title('Combined PCA Results', fontsize=16)
            plt.legend(handles=handles, labels=labels, title='Legend', fontsize=12)

        plt.xlabel(f"PC1: {explained_variance_ratio[0]:.2%} explained variance", fontsize=14)
        plt.ylabel(f"PC2: {explained_variance_ratio[1]:.2%} explained variance", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def pcoa_analyze(data: pd.DataFrame, n_components: int = 2,  subject_column: str = None, individual: bool = True,
                     base_color: str = 'blue') -> None:

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("Input 'data' DataFrame is empty.")

        if subject_column and subject_column not in data.columns:
            raise ValueError(f"'subject_column' '{subject_column}' not found in input DataFrame.")

        # Check if the subject_column exists in the DataFrame
        subject = None
        if subject_column in data.columns:
            subject = data[subject_column].values
            data = data.drop([subject_column], axis=1)

        # Calculate CLR transformation
        clr_concatenated_df = clr(data + 1)

        # Calculate normalized Aitchinson distance matrix
        clr_concatenated_distance_matrix = PCoACalculationAndPlots.create_normalized_aitchinson_distance_matrix(clr_concatenated_df)

        # Run PCA on the normalized Aitchinson distance matrix
        pca_results, exp_var = PCoACalculationAndPlots.run_pca(clr_concatenated_distance_matrix, n_components, subject)

        # Visualize the first two components
        PCoACalculationAndPlots.visualize_pca(pca_results, exp_var, individual, base_color)
