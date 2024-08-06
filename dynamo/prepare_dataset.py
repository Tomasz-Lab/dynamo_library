import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate


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


