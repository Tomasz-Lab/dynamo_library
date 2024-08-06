import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VisualizeTaxonomy:

    @staticmethod
    def prepare_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
        DF = pd.DataFrame()
        for i in range(2, 8):
            split_taxonomy = df.Taxon.str.split('; ', expand=True).iloc[:, :i]
            phylum = split_taxonomy[split_taxonomy.columns[0:]].apply(lambda x: '; '.join(x.dropna().astype(str)),
                                                                      axis=1).values
            DF[i] = phylum

        DF.columns = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Strain']
        DF['feature'] = df['Feature ID']

        return DF

    @staticmethod
    def filter_taxonomy(input_taxonomy_df: pd.DataFrame, counts_df: pd.DataFrame, level: str, threshold: int) -> pd.DataFrame:

        taxonomy_df = VisualizeTaxonomy.prepare_taxonomy(input_taxonomy_df)
        sequence_taxonomy_dict = dict(zip(taxonomy_df.feature, taxonomy_df[level]))
        renamed_counts_df = counts_df.rename(columns=sequence_taxonomy_dict)

        mean_df = renamed_counts_df.T.reset_index().groupby(by=['index']).sum().mean(axis=1).reset_index().sort_values(
            by=[0], ascending=False).reset_index(drop=True)

        # rename bacteria
        mean_df['name'] = np.where(mean_df['index'].isin(mean_df.iloc[:threshold]['index']), mean_df['index'], 'other')

        # create new dict of bacteria names
        new_level_dict = dict(zip(mean_df['index'], mean_df['name']))
        renamed_counts_df = renamed_counts_df.rename(columns=new_level_dict)
        renamed_counts_df = renamed_counts_df.T.reset_index().groupby(['index']).sum().T

        # change to relative abundance
        rel_ab_df = renamed_counts_df.div(renamed_counts_df.sum(axis=1), axis=0)

        return rel_ab_df

    @staticmethod
    def plot_stacked_taxonomy(df: pd.DataFrame, level: str, subject: str) -> None:
        colors = ['#ee9b00', '#588157', '#ca6702',
                  '#9b2226', '#e9c46a', '#2a9d8f',
                  '#264653', '#560bad', '#e76f51',
                  '#54478c', '#2c699a', '#3c096c', '#e9d8a6']

        kwargs = {'alpha': 1}
        ax = df.plot(kind='bar', stacked=True, width=1, color=colors, figsize=(20, 3), **kwargs)
        ax.set_title(f"Taxonomy visualization for {subject}", fontsize=16)

        plt.legend(bbox_to_anchor=(1, 1), title=f'{level} \n', ncol=1, edgecolor='white')

        plt.xticks(np.arange(0, (len(df) + 1), 100), fontsize=8, rotation=360)
        plt.yticks(fontsize=8)
        plt.xlabel('time point [day]', size=10)
        plt.ylabel('relative abundance', size=10)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()
        return ax

