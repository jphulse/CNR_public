import pyagrum as gum
import pandas as pd
import os
import sys
import time
from datetime import datetime
import numpy as np


# Saves synthetic data to csv folders
def save_synthetic_data(df_100: pd.DataFrame, df_500: pd.DataFrame, df_1000: pd.DataFrame, adjacency_matrix: np.ndarray) ->None:
    """Saves the synthetic dataframes to properly named files which are of the type year_month_day_hour_minute_secondN.csv

    Args:
        df_100 (pd.DataFrame): dataframe with 100 rows representative of the network
        df_500 (pd.DataFrame): dataframe with 500 rows representative of the network
        df_1000 (pd.DataFrame): dataframe with 1000 rows representative of the network
        adjacency_matrix (np.ndarray): adjacency matrix representation of the network
    """
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    directory = os.path.join('data', 'synthetic', timestamp)
    os.makedirs(directory, exist_ok=True)
    files = {
        '100.csv': df_100,
        '500.csv': df_500,
        '1000.csv': df_1000
    }
    for filename, df in files.items():
        df.to_csv(os.path.join(directory, filename), header=False, index=False)
    np.savetxt(os.path.join(directory, 'truth.csv'), adjacency_matrix, delimiter=',', fmt='%s')




def main(N=20, domain_size = 10, vert_to_edge_rati=2.0) -> None:
    """Main method generates synthetic bayes net with N variables, with the paramatized domain size and density, prints them out to files

    Args:
        N (int, optional): number of nodes in the graph. Defaults to 20.
        domain_size (int, optional): size of the domain of each variable in the network. Defaults to 10.
        vert_to_edge_rati (float, optional): number of edges to vertices. Defaults to 2.0.
    """
    bn = gum.randomBN(n=N, domain_size=domain_size, ratio_arc=vert_to_edge_rati)
    generator_1000 = gum.BNDatabaseGenerator(bn)
    generator_500 = gum.BNDatabaseGenerator(bn)
    generator_100 = gum.BNDatabaseGenerator(bn)
    generator_1000.drawSamples(1000)
    generator_500.drawSamples(500)
    generator_100.drawSamples(100)
    df_1000 = generator_1000.to_pandas()
    df_500 = generator_500.to_pandas()
    df_100 = generator_100.to_pandas()
    adj_mat = bn.adjacencyMatrix()
    save_synthetic_data(df_100, df_500, df_1000, adj_mat)

if __name__ == '__main__':
    x = 0
    try:
       x = int(sys.argv[1])
    except:
       x = 1
    
    for _ in range(x):
        main()
        time.sleep(1)
    
    
    
    
    
