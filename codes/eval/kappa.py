import numpy as np
import pandas as pd

import statsmodels
from statsmodels.stats.inter_rater import fleiss_kappa
from statistics import mean

def get_fleiss_kappa(datasets):

    results = {}

    # print(datasets)

    ## extract corresponding columns from the three datasets
    for col in datasets[0].columns[1:]:
        dataset_columns = []
        for dataset in datasets:
            dataset_columns.append(dataset[col])
            # print(dataset[col])

        ## find fleiss kapa values for corresponding columns
        agreement_matrix = pd.concat(dataset_columns, axis = 1)
        # print(agreement_matrix)
        table = statsmodels.stats.inter_rater.aggregate_raters(agreement_matrix.values)
        print(table[0])

        res = fleiss_kappa(table[0])
        results[col] = res

        print(res)


    return results


files = [pd.read_csv(file) for file in ["anno1.csv", "anno2.csv", "anno3.csv"]]

results1 = get_fleiss_kappa(files)
print(results1)
