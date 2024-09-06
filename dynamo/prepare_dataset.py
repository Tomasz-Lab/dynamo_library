import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate


class PrepareDataset:
    @staticmethod
    def interpolate_pchip(df: pd.DataFrame, path: str, subject: str) -> pd.DataFrame:
        def prepare_data_for_interpolation(df: pd.DataFrame) -> pd.DataFrame:
            start_df = df.iloc[0].name
            end_df = df.iloc[-1].name

            full = list(range(start_df, end_df))
            missing_t_points = list(set(full) - set(df.index.astype(int)))
            missing_df = df.reindex(df.index.union(missing_t_points))

            return missing_df

        def pchip_interpolation(col: str, masked_df: pd.DataFrame) -> pd.DataFrame:
            df_interpolated = pd.DataFrame(index=masked_df.index)
            tmp = masked_df[col]
            base_nodes = tmp.dropna().index
            interpolated_nodes = tmp[tmp.isna()].index
            y = pchip_interpolate(base_nodes, tmp.dropna().values, interpolated_nodes)
            name = str(col)
            df_interpolated.loc[base_nodes, name] = tmp.dropna().values
            df_interpolated.loc[interpolated_nodes, name] = y
            return df_interpolated

        def apply_interpolation(df: pd.DataFrame, interpolation_function) -> pd.DataFrame:
            interpolated_columns = []
            for col in df.columns:
                interpolated_col = interpolation_function(col, df)
                interpolated_columns.append(interpolated_col)
            interpolated_df = pd.concat(interpolated_columns, axis=1)
            return interpolated_df

        df = prepare_data_for_interpolation(df)
        df_interpolated = apply_interpolation(df, pchip_interpolation)
        df_interpolated = df_interpolated.astype(int).T
        df_interpolated.to_csv(path + f'{subject}_interpolated.tsv', sep='\t')
        return df_interpolated

    @staticmethod
    def rarefy_df(df, depth):

        rarefied_df = pd.DataFrame(index=df.index, columns=df.columns)

        for col in df.columns:
            col_data = df[col].values
            total_reads = col_data.sum()

            if total_reads <= depth:
                # If total reads are less than or equal to depth, use the entire column
                rarefied_df[col] = col_data
            else:
                indices = np.repeat(np.arange(len(col_data)), col_data)
                sampled_indices = np.random.choice(indices, size=depth, replace=False)

                # Count occurrences of each index in the sampled data
                counts = np.bincount(sampled_indices, minlength=len(col_data))
                rarefied_df[col] = counts

        return rarefied_df

    @staticmethod
    def plot_alpha_rarefaction(datasets, dataset_labels, sampling_depth=20000, step=100):

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        best_depth = None
        max_observed_features = -np.inf

        for i, df in enumerate(datasets):
            observed_features = []
            samples_retained = []

            for depth in range(step, sampling_depth + 1, step):
                rarefied_df = PrepareDataset.rarefy_df(df, depth)

                observed_features.append((rarefied_df > 0).sum().sum())

                # Number of samples retained: count columns (time points) that still have non-zero values after rarefaction
                retained_samples = (rarefied_df > 0).any(axis=0).sum()
                samples_retained.append(retained_samples)

                # Update best depth (use the one that retains most features)
                if observed_features[-1] > max_observed_features:
                    max_observed_features = observed_features[-1]
                    best_depth = depth

            # Plot observed features
            axes[0].plot(range(step, sampling_depth + 1, step), observed_features, label=dataset_labels[i])

            # Plot number of samples retained
            axes[1].plot(range(step, sampling_depth + 1, step), samples_retained, label=dataset_labels[i])

        # Mark best sequencing depth on both plots
        for ax in axes:
            ax.axvline(best_depth, color='red', linestyle='--', label=f'Optimal Depth: {best_depth}')
            ax.legend()

        axes[0].set_title('Observed Features vs. Sampling Depth')
        axes[0].set_xlabel('Sampling Depth')
        axes[0].set_ylabel('Observed Features')
        axes[0].grid(True)

        axes[1].set_title('Number of Samples Retained vs. Sampling Depth')
        axes[1].set_xlabel('Sampling Depth')
        axes[1].set_ylabel('Number of Samples Retained')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

        return best_depth
