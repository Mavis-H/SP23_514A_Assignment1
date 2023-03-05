# Author: Siqian Hou
# Date: 03/02/2023
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt


# Create standardized data into a new xls file.
def create_standardized_data():
    _raw_data_df = pd.read_excel("data/Concrete_Data.xls")
    _standardization_data = preprocessing.scale(_raw_data_df)
    _standardization_data_df = pd.DataFrame(_standardization_data)
    _standardization_data_df.to_excel(excel_writer="data/Standardization_Data.xls", startrow=1, index=False,
                                      header=False)


# Generate distribution histogram figures for both raw data and standardized data.
def create_data_distribution_figures(_raw_data_df, _standardization_data_df):
    raw_ax = _raw_data_df.plot.hist(alpha=0.3, title="Raw Data")
    standardization_ax = _standardization_data_df.plot.hist(alpha=0.3, title="Standardized Data")
    plt.show()
