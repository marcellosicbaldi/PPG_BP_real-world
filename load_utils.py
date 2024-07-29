import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr, importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

def load_axivity(filename: str, sampling_rate: int = 100):
    """
    Load Axivity data from a .cwa file.

    Parameters
 
    filename : str
        The path to the .cwa file.

    sampling_rate : int
        The desired sampling rate of the output (data will be interpolated). Default is 100 Hz.

    Returns

    acc_data : pd.DataFrame
        A DataFrame containing the data from the .cwa file.
    """

    utils = importr('utils')
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)
    pandas2ri.activate()
    GGIRread = importr("GGIRread")

    lw_dummy = GGIRread.readAxivity(filename, start = 0, end = 0)
    header = dict(zip(lw_dummy.names, list(lw_dummy)))["header"]
    numBlocks_dummy = dict(zip(header.names, list(header)))
    numBlocks = int(np.asanyarray(numBlocks_dummy["blocks"]))

    data = ro.conversion.rpy2py(dict(zip(GGIRread.readAxivity(filename, start = 0, end = numBlocks).names, list(GGIRread.readAxivity(filename, start = 0, end = numBlocks))))["data"])
    acc_data = data[["x", "y", "z"]]
    acc_data.index = pd.to_datetime(data['time'], unit = 's') + pd.Timedelta(hours=1)
    return acc_data

def load_csv(filename: str, E4):
    """
    Load a .csv file.

    Parameters
    filename : str
        The path to the .csv file.

    Returns
    acc_data : pd.DataFrame
        A DataFrame containing the data from the .csv file.
    """

    if E4:
        data = pd.read_csv(filename, header = None)
        timestamp_start = data.iloc[0, 0].astype(int)
        acc_data = data.iloc[2:] / 64
        timestamps = np.arange(0, len(acc_data)/32, 1/32) + timestamp_start
        acc_data.index = pd.to_datetime(timestamps, unit = 's')
        acc_data.columns = ["x", "y", "z"]
    else:
        acc_data = pd.read_csv(filename)
    
    return acc_data

def load_pkl(filename: str):
    """
    Load a .pkl file.

    Parameters
    filename : str
        The path to the .pkl file.

    Returns
    data : pd.DataFrame
        A DataFrame containing the data from the .pkl file.
    """

    acc_data = pd.read_pickle(filename)
    return acc_data

def load_geneactiv(filename: str):
    """ Load a .bin file from the GENEActiv.

    Parameters
    filename : str
        The path to the .bin file.

    Returns
    acc_data : pd.DataFrame
        A DataFrame containing the data from the .bin file.
    """

    pass
