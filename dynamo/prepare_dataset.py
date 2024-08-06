import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.stats import sem


class PrepareDataset:
    """
    A class to prepare datasets for analysis, including interpolation of missing values
    and rarefaction of samples.
    """

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
    def rarefy_table(df: pd.DataFrame, max_depth: int) -> tuple[pd.DataFrame, dict]:

        def rarefy_sample(sample, depth):

            total_counts = sample.sum()
            if total_counts < depth:
                # If the total is less than the desired depth, return the original sample
                return sample.values

            if total_counts == 0:
                # If the total is zero, return zeros
                return np.zeros_like(sample.values)

            # Calculate probabilities
            pvals = sample / total_counts

            # Remove NaNs and ensure probabilities are valid
            pvals = np.nan_to_num(pvals, nan=0.0)

            if np.any(pvals < 0) or np.any(pvals > 1):
                raise ValueError("Invalid probabilities: pvals must be in the range [0, 1].")

            # Perform multinomial sampling
            return np.random.multinomial(depth, pvals)

        rarefaction_curves = {}

        rarefied_table = df.apply(lambda x: rarefy_sample(x, max_depth), axis=1)

        for idx, row in df.iterrows():
            richness = []
            for depth in range(1, max_depth + 1):
                rarefied_sample = rarefy_sample(row, depth)
                richness.append(np.count_nonzero(rarefied_sample))
            rarefaction_curves[idx] = richness

        rarefied_table = pd.DataFrame(rarefied_table.tolist(), index=df.index, columns=df.columns)

        return rarefied_table, rarefaction_curves

    @staticmethod
    def find_optimal_depth(rarefaction_curves: dict) -> int:

        optimal_depths = []
        for idx, richness in rarefaction_curves.items():

            changes = np.diff(richness)
            # Find the depth where the change is minimal
            optimal_depth = np.argmin(changes) + 1  # +1 because np.diff reduces the length by 1
            optimal_depths.append(optimal_depth)
        return int(np.median(optimal_depths))





