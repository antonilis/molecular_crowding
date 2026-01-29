import os
import pandas as pd

import utils as uts
import uncertainties as unc
from uncertainties import umath
import numpy as np

import matplotlib.pyplot as plt

from source_data_processors.reacting_probes_equillibrium_constants import EquillibriumConstants
from source_data_processors.sodium_crowder_complexation_constant import SodiumComplexation

class Data:

    def __init__(self, source_equillbrium_constants_path, source_complexation_data_path ):

        self.EquillbriumConstantsObj = EquillibriumConstants(source_equillbrium_constants_path)
        self.SodiumCrowderComplexationObj = SodiumComplexation(source_complexation_data_path)

        self.data = self.merge_data()



    def merge_data(self):

        eq_data = self.EquillbriumConstantsObj.reactions_equillibrium_constants

        eq_data['n Beta [1/M]'] = eq_data['No_mono'] * self.SodiumCrowderComplexationObj.n_complexation


        return eq_data



if __name__ == '__main__':

    equillibrium_constans_path = 'source_data\\MacromoleculeEquilibria_Crowding'
    complexation_data_path = 'source_data\\IonCrowderComplexation\\raw_data'

    data = Data(equillibrium_constans_path, complexation_data_path)




    # source_data_path = '.\\source_data\MacromoleculeEquilibria_Crowding'  # source_data\MacromoleculeEquilibria_Crowding
    #
    # obj = Data(source_data_path)
    # df = obj.calculate_parabolic_fit()
    # groups = df[['source', 'sample', 'crowder']].drop_duplicates()

    # for _, row in groups.iterrows():
    #     mask = (
    #             (df['source'] == row['source']) &
    #             (df['sample'] == row['sample']) &
    #             (df['crowder'] == row['crowder'])
    #     )
    #     df_group = df[mask]
    #     group_name = f"{row['source']}_{row['sample']}_{row['crowder']}"
    #
    #     plot_fit(df_group, group_name)
    #     plt.savefig(f'.\\plots\\{group_name}.png')
    #     plt.close()

    # zostaw tylko to co jest dobre (znaki liccb i dofitowanie) problem ze stężeniem
