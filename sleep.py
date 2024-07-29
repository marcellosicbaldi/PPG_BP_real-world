import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 0.96
import time
import warnings
warnings.filterwarnings("ignore")
from load_utils import * 

def sleep(filename: str, sampling_rate: int = 100, method: str = "vanhees2015", E4 = False, plot_bool = False):
	"""
	Estimate sleep from raw acceleration data.

	Parameters
	filename : str
		The full path and filename of the raw acceleration data.

	Returns
	sleep_periods : pd.DataFrame
		A DataFrame containing the start and end times of the estimated sleep periods.
	"""

	# start = time.time()

	print("Loading data")

	if filename.lower().endswith(".cwa"):
		acc_data = load_axivity(filename, sampling_rate)
	elif filename.endswith(".pkl"):
		acc_data = load_pkl(filename)
	elif filename.endswith(".csv"):
		acc_data = load_csv(filename, E4 = E4)
		
	print("Data loaded")

	# Non-wear time detection
	nonwear_time = _hees_2013_calculate_non_wear_time(acc_data.values, hz = sampling_rate)
	acc_data["non_wear"] = nonwear_time

	# Sleep detection
	if method == "vanhees2015":
		sleep_periods = _vanhees2015(acc_data, plot_bool = plot_bool)
	elif method == "Cole-Kripke":
		sleep_periods = _cole_kripke(acc_data)
	elif method == "Sadeh":
		sleep_periods = _sadeh(acc_data)
	elif method == "sundararajan":
		sleep_periods = _sundararajan(acc_data.drop(columns = ["non_wear"]))
			
	return sleep_periods


def _vanhees2015(acc_data, plot_bool):
	"""Van Hees 2015 sleep detection.

	Args:
		acc_data : pd.DataFrame
			The raw accelerometer data.

	Returns:
		sleep_periods : pd.DataFrame
			A DataFrame containing the start and end times of the estimated sleep periods.

	"""

	# start = time.time()

	# 2. Estimate sleep periods
	# z angle calculation
	z1_angle = np.arctan(acc_data['z'].rolling('5 s').mean() /
						acc_data['x'].rolling('5 s').mean()**2 + acc_data['y'].rolling('5 s').mean()**2) * 180 / np.pi

	# averaging the z-angle every 5 consecutive seconds
	z_angle_consmean = z1_angle.resample('5 s').mean()

	non_wear_res = acc_data['non_wear'].resample('5 s').max()

	# absolute differences between successive 5 seconds averages
	z_angle_diff = z_angle_consmean.diff().bfill().abs()

	# 5-minute moving median (there are 60 five seconds interval in 5 minutes)
	z_angle = z_angle_diff.rolling('5 min').median()

	z_angle = pd.DataFrame(z_angle, columns = ["z_angle"])
	z_angle["non_wear"] = non_wear_res

	# mask non-wear time
	z_angle.mask(z_angle["non_wear"] == 0, inplace = True)
	z_angle.drop(columns = ["non_wear"], inplace = True)

	th = np.percentile(z_angle.dropna(), 10) * 15
	sleep1 = (z_angle["z_angle"] < th).astype(int)

	start_sleep_blocks = sleep1.where(sleep1.diff()==1).dropna()
	end_sleep_blocks = sleep1.where(sleep1.diff()==-1).dropna()

	if sleep1.iloc[0] == 1:
		start_sleep_blocks = pd.concat([pd.Series(0, index = [sleep1.index[0]]), start_sleep_blocks])
	if sleep1.iloc[-1] == 1:
		end_sleep_blocks = pd.concat([end_sleep_blocks, pd.Series(0, index = [sleep1.index[-1]])])

	duration_sleep_blocks = pd.DataFrame({"duration": end_sleep_blocks.index - start_sleep_blocks.index}, index = start_sleep_blocks.index)
	long_sleep_blocks = duration_sleep_blocks[duration_sleep_blocks["duration"] > pd.Timedelta("30 min")]
	long_sleep_blocks["label"] = 1 # sleep

	start_sleep = long_sleep_blocks.index
	end_sleep = pd.to_datetime((long_sleep_blocks.index + long_sleep_blocks["duration"]).values)

	end_sleep = end_sleep.to_series().reset_index(drop = True)
	start_sleep = start_sleep.to_series().reset_index(drop = True)

	duration_between_sleep_blocks = (start_sleep.iloc[1:].values - end_sleep.iloc[:-1].values)

	for i in range(len(start_sleep)-1):
		if duration_between_sleep_blocks[i] < pd.Timedelta("60 min"):
			end_sleep[i] = np.nan
			start_sleep[i+1] = np.nan
	end_sleep.dropna(inplace = True)
	start_sleep.dropna(inplace = True)

	# end = time.time()
	# print(f"Time elapsed: {end - start} seconds")

	# plot sleep periods using axvspan
	if plot_bool:
		plt.figure(figsize=(10, 5))
		plt.plot(z_angle["z_angle"])
		for i in range(len(start_sleep)):
			plt.axvspan(start_sleep.iloc[i], end_sleep.iloc[i], color='r', alpha=0.3)
		plt.axhline(th, color='r', linestyle='--')
			
	for i in range(len(start_sleep)):
		print(f"Sleep period {i+1} starts at {start_sleep.iloc[i]} and ends at {end_sleep.iloc[i]}")

	# Save start and end sleep times to a csv file
	sleep_periods = pd.DataFrame({"start": start_sleep.values, "end": end_sleep.values})

	return sleep_periods

def _cole_kripke(acc_data, threshold = 1, rescoring = False, plot_bool = False):
	"""Cole-Kripke sleep detection.
	Parameters
	----------
	acc_data : pd.DataFrame
		The raw accelerometer data.
		
	Returns
	-------
	sleep_periods : pd.DataFrame
		A DataFrame containing the start and end times of the estimated sleep periods.
	"""
	rs_f = 30  # 60sec/2sec
	# Define the scale and weights for this settings
	scale = 0.001
	window = np.array(
		[106, 54, 58, 76, 230, 74, 67, 0, 0],
		np.int32
	)

	ck = acc_data.rolling(
        window.size, center=True
    ).apply(window_convolution, args=(scale, window), raw=True)
	
	return (ck < threshold).astype(int)


def _sadeh(acc_data):
	"""Sadeh sleep detection.
	Parameters
	----------
	acc_data : pd.DataFrame
		The raw accelerometer data.
		
	Returns
	-------
	sleep_periods : pd.DataFrame
		A DataFrame containing the start and end times of the estimated sleep periods.
	"""
	pass

def _sundararajan(acc_data):
      """Sundararajan sleep detection.
	  Parameters
	  ----------
	  acc_data : pd.DataFrame
		  The raw accelerometer data.
			
	  Returns
	  -------
	  sleep_periods : pd.DataFrame
		  A DataFrame containing the start and end times of the estimated sleep periods.
	  """
	  # Seemed promising, but uses an ensemble of RF models which are >500MB each... can't use it

def _hees_2013_calculate_non_wear_time(data, hz, min_non_wear_time_window = 30, window_overlap = 15, std_mg_threshold = 0.08, std_min_num_axes = 2 , value_range_mg_threshold = 0.2, value_range_min_num_axes = 2):
	"""
	Estimation of non-wear time periods based on Hees 2013 paper
      taken from https://github.com/shaheen-syed/ActiGraph-ActiWave-Analysis/tree/master/algorithms/non_wear_time

	Estimation of Daily Energy Expenditure in Pregnant and Non-Pregnant Women Using a Wrist-Worn Tri-Axial Accelerometer
	Vincent T. van Hees  , Frida Renström , Antony Wright, Anna Gradmark, Michael Catt, Kong Y. Chen, Marie Löf, Les Bluck, Jeremy Pomeroy, Nicholas J. Wareham, Ulf Ekelund, Søren Brage, Paul W. Franks
	Published: July 29, 2011https://doi.org/10.1371/journal.pone.0022922

	Accelerometer non-wear time was estimated on the basis of the standard deviation and the value range of each accelerometer axis, calculated for consecutive blocks of 30 minutes. 
	A block was classified as non-wear time if the standard deviation was less than 3.0 mg (1 mg = 0.00981 m·s−2) for at least two out of the three axes or if the value range, for 
	at least two out of three axes, was less than 50 mg.

	Parameters
	----------
	data: np.array(n_samples, axes)
		numpy array with acceleration data in g values. Each column represent a different axis, normally ordered YXZ
	hz: int (optional)
		sample frequency in hertz. Indicates the number of samples per 1 second. Default to 100 for 100hz. The sample frequency is necessary to 
		know how many samples there are in a specific window. So let's say we have a window of 15 minutes, then there are hz * 60 * 15 samples
	min_non_wear_time_window : int (optional)
		minimum window length in minutes to be classified as non-wear time
	window_overlap : int (optional)
		basically the sliding window that progresses over the acceleration data. Defaults to 15 minutes.
	std_mg_threshold : float (optional)
		standard deviation threshold in mg. Acceleration axes values below or equal this threshold can be considered non-wear time. Defaults to 3.0g. 
		Note that within the code we convert mg to g.
	std_min_num_axes : int (optional) 
		minimum numer of axes used to check if acceleration values are below the std_mg_threshold value. Defaults to 2 axes; meaning that at least 2 
		axes need to have values below a threshold value to be considered non wear time
	value_range_mg_threshold : float (optional)
		value range threshold value in mg. If the range of values within a window is below this threshold (meaning that there is very little change 
		in acceleration over time) then this can be considered non wear time. Default to 50 mg. Note that within the code we convert mg to g
	value_range_min_num_axes : int (optional)
		minimum numer of axes used to check if acceleration values range are below the value_range_mg_threshold value. Defaults to 2 axes; meaning that at least 2 axes need to have a value range below a threshold value to be considered non wear time

	Returns
	---------
	non_wear_vector : np.array((n_samples, 1))
		numpy array with non wear time encoded as 0, and wear time encoded as 1.
	"""

	# number of data samples in 1 minute
	num_samples_per_min = hz * 60

	# define the correct number of samples for the window and window overlap
	min_non_wear_time_window *= num_samples_per_min
	window_overlap *= num_samples_per_min

	# convert the standard deviation threshold from mg to g
	std_mg_threshold /= 1000
	# convert the value range threshold from mg to g
	value_range_mg_threshold /= 1000

	# new array to record non-wear time. Convention is 0 = non-wear time, and 1 = wear time. Since we create a new array filled with ones, we only have to 
	# deal with non-wear time (0), since everything else is already encoded as wear-time (1)
	non_wear_vector = np.ones((data.shape[0], 1), dtype = 'uint8')

	# loop over the data, start from the beginning with a step size of window overlap
	for i in range(0, len(data), window_overlap):

		# define the start of the sequence
		start = i
		# define the end of the sequence
		end = i + min_non_wear_time_window

		# slice the data from start to end
		subset_data = data[start:end]

		# check if the data sequence has been exhausted, meaning that there are no full windows left in the data sequence (this happens at the end of the sequence)
		# comment out if you want to use all the data
		if len(subset_data) < min_non_wear_time_window:
			break

		# calculate the standard deviation of each column (YXZ)
		std = np.std(subset_data, axis=0)

		# check if the standard deviation is below the threshold, and if the number of axes the standard deviation is below equals the std_min_num_axes threshold
		if (std < std_mg_threshold).sum() >= std_min_num_axes:

			# at least 'std_min_num_axes' are below the standard deviation threshold of 'std_min_num_axes', now set this subset of the data to 0 which will 
			# record it as non-wear time. Note that the full 'new_wear_vector' is pre-populated with all ones, so we only have to set the non-wear time to zero
			non_wear_vector[start:end] = 0

		# calculate the value range (difference between the min and max) (here the point-to-point numpy method is used) for each column
		value_range = np.ptp(subset_data, axis = 0)

		# check if the value range, for at least 'value_range_min_num_axes' (e.g. 2) out of three axes, was less than 'value_range_mg_threshold' (e.g. 50) mg
		if (value_range < value_range_mg_threshold).sum() >= value_range_min_num_axes:

			# set the non wear vector to non-wear time for the start to end slice of the data
			# Note that the full array starts with all ones, we only have to set the non-wear time to zero
			non_wear_vector[start:end] = 0

	return non_wear_vector