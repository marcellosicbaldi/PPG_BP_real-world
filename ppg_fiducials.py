import numpy as np
import pandas as pd
from scipy.signal import find_peaks, filtfilt, firls
import copy

# All of these functions were adapted from the pyPPG library available at https://github.com/godamartonaron/GODA_pyPPG
# Goda, M. A., Charlton, P. H., & Behar, J. A. (2023). pyPPG: A Python toolbox for comprehensive photoplethysmography signal analysis. DOI 10.1088/1361-6579/ad33a2

def get_dicrotic_notch(ppg: np.array, fs, peaks: np.array, onsets: list):
    """
    Dicrotic Notch function estimate the location of dicrotic notch in between the systolic and diastolic peak

    :param peaks: peaks of the signal
    :type peaks: 1-d array
    :param onsets: onsets of the signal
    :type onsets: list

    :return: location of dicrotic notches, 1-d array
    """

    ## The 2nd derivative and Hamming low pass filter is calculated.
    dxx = np.diff(np.diff(ppg))

    # # Make filter
    # Fn = fs / 2                                 # Nyquist Frequency
    # FcU = 20                                    # Cut off Frequency: 20 Hz
    # FcD = FcU + 5                               # Transition Frequency: 5 Hz

    # n = 21                                      # Filter order
    # f = [0, (FcU / Fn), (FcD / Fn), 1]          # Frequency band edges
    # a = [1, 1, 0, 0]                            # Amplitudes
    # b = firls(n, f, a)

    # lp_ppg = filtfilt(b, 1,  dxx)          # Low pass filtered signal with 20 cut off Frequency and 5 Hz Transition width

    lp_ppg = dxx # skip LP filter, since using the interpolated pulses and dont want to make a mess. Moreover, Empatica E4 PPG is already filtered.

    ## The weighting is calculated and applied to each beat individually
    def t_wmax(i, peaks, onsets):
        if i < 3:
            HR = np.mean(np.diff(peaks))/fs
            t_wmax = -0.1 * HR + 0.45
        else:
            t_wmax = np.mean(peaks[i - 3:i]-onsets[i - 3:i])/fs
        return t_wmax

    dic_not=[]
    for i in range(0,len(onsets)-1):
        nth_beat = lp_ppg[onsets[i]:onsets[i + 1]]

        i_Pmax=peaks[i]-onsets[i]
        t_Pmax=(peaks[i]-onsets[i])/fs
        t=np.linspace(0,len(nth_beat)-1,len(nth_beat))/fs
        T_beat=(len(nth_beat)-1)/fs
        tau=(t-t_Pmax)/(T_beat-t_Pmax)
        tau[0:i_Pmax] = 0
        beta=5

        if len(peaks)>1:
            t_w=t_wmax(i, peaks, onsets)
        else:
            t_w=np.NaN

        if t_w!=T_beat:
            tau_wmax=(t_w-t_Pmax)/(T_beat-t_Pmax)
        else:
            tau_wmax=0.9

        alfa=(beta*tau_wmax-2*tau_wmax+1)/(1-tau_wmax)
        if (alfa > 4.5) or (alfa < 1.5):
            HR = np.mean(np.diff(peaks))/fs
            t_w = -0.1 * HR + 0.45
            tau_wmax = (t_w - t_Pmax) / (T_beat - t_Pmax)
            alfa = (beta * tau_wmax - 2 * tau_wmax + 1) / (1 - tau_wmax)

        ## Calculate the Dicrotic Notch for each heart cycle using the weighted window
        if alfa>1:
            w = tau ** (alfa - 1) * (1 - tau) ** (beta - 1)
        else:
            w = tau * (1 - tau) ** (beta - 1)

        pp=w*nth_beat
        pp = pp[np.where(~np.isnan(pp))]
        max_pp_v = np.max(pp)
        max_pp_i=np.where(pp==max_pp_v)[0][0]
        ## NOTE!! Shifting with 36 ms. -- Determined empirically -- see: https://github.com/godamartonaron/GODA_pyPPG/issues/12
        shift=int(fs*0.036)
        dic_not.append(max_pp_i+onsets[i]+shift)

    return dic_not

def get_apg_fiducials(ppg, onsets: list, peaks: np.array):
        """Calculate Second derivitive points a, b, c, d, e, and f from the PPG" signal

        :param onsets: onsets of the signal
        :type onsets: list
        :param peaks: peaks of the signal
        :param types: 1-d array

        :return:
            - a: The highest amplitude between pulse onset and systolic peak on PPG"
            - b: The first local minimum after the a-point on PPG"
            - c: The local maximum with the highest amplitude between the b-point and e-point, or if no local maximum is present then the inflection point on PPG"
            - d: The local minimum with the lowest amplitude between the c-point and e-point, or if no local minimum is present then the inflection point on PPG"
            - e: The local maximum with the highest amplitude after the b-point and before the diastolic peak on PPG"
            - f: The first local minimum after the e-point on PPG"
        """

        sig = ppg
        ddx = np.gradient(ppg)
        dddx = np.gradient(ppg)

        nan_v = np.empty(len(onsets)-1)
        nan_v[:] = np.NaN
        a, b, c, d, e, f = copy.deepcopy(nan_v),copy.deepcopy(nan_v),copy.deepcopy(nan_v),copy.deepcopy(nan_v),copy.deepcopy(nan_v),copy.deepcopy(nan_v)
        for i in range(0,len(onsets)-1):

            try:
                # a fiducial point
                temp_pk=np.argmax(sig[onsets[i]:onsets[i + 1]])+onsets[i]-1
                temp_segment=ddx[onsets[i]:temp_pk]
                max_locs, _ = find_peaks(temp_segment)
                try:
                    max_loc = max_locs[np.argmax(temp_segment[max_locs])]
                except:
                    max_loc = temp_segment.argmax()

                max_a = max_loc + onsets[i] - 1
                a[i] = max_a

                # b fiducial point
                temp_segment=ddx[int(a[i]):onsets[i+1]]
                min_locs, _ = find_peaks(-temp_segment)
                min_b = min_locs[0] + a[i] - 1
                b[i] = min_b

                # e fiducial point
                e_lower_bound = peaks[i]
                upper_bound_coeff = 0.6
                e_upper_bound = ((onsets[i + 1] - onsets[i]) * upper_bound_coeff + onsets[i]).astype(int)
                temp_segment=ddx[int(e_lower_bound):int(e_upper_bound)]
                max_locs, _ = find_peaks(temp_segment)
                if max_locs.size==0:
                    temp_segment=ddx[int(peaks[i]):onsets[i + 1]]
                    max_locs, _ = find_peaks(temp_segment)

                try:
                    max_loc = max_locs[np.argmax(temp_segment[max_locs])]
                except:
                    max_loc = temp_segment.argmax()

                max_e = max_loc + e_lower_bound - 1
                e[i] = max_e

                # c fiducial point
                temp_segment = ddx[int(b[i]):int(e[i])]
                max_locs, _ = find_peaks(temp_segment)
                if max_locs.size>0:
                    max_loc = max_locs[0]
                else:
                    temp_segment = dddx[int(b[i]):int(e[i])]
                    min_locs, _ = find_peaks(-temp_segment)

                    if min_locs.size > 0:
                        max_loc = min_locs[np.argmin(temp_segment[min_locs])]
                    else:
                        max_locs, _ = find_peaks(temp_segment)
                        try:
                            max_loc = max_locs[0]
                        except:
                            max_loc = temp_segment.argmax()

                max_c = max_loc + b[i] - 1
                c[i] = max_c

                # d fiducial point
                temp_segment = ddx[int(c[i]):int(e[i])]
                min_locs, _ = find_peaks(-temp_segment)
                if min_locs.size > 0:
                    min_loc = min_locs[np.argmin(temp_segment[min_locs])]
                    min_d = min_loc + c[i] - 1
                else:
                    min_d = max_c

                d[i] = min_d

                # f fiducial point
                temp_segment = ddx[int(e[i]):onsets[i + 1]]
                min_locs, _ = find_peaks(-temp_segment)
                if (min_locs.size > 0) and (min_locs[0]<len(sig)*0.8):
                    min_loc = min_locs[0]
                else:
                    min_loc = 0

                min_f = min_loc + e[i] - 1
                f[i] = min_f
            except:
                pass

        apg_fp = pd.DataFrame({"a":[], "b":[], "c":[],"d":[], "e":[], "f":[]})
        apg_fp.a, apg_fp.b, apg_fp.c, apg_fp.d, apg_fp.e, apg_fp.f = a, b, c, d, e, f
        return apg_fp

def get_diastolic_peak(ppg: np.array, diag_pulses: list, dicroticnotch: list, e_point: pd.Series):
        """
        Diastolic peak function -- estimates the location of diastolic peak in between the dicrotic notch and e-point

        :param onsets: onsets of the signal
        :type onsets: list
        :param dicroticnotches: dicrotic notches of the signal
        :type dicroticnotches: list
        :param e_point: e-points of the signal
        :type e_point: pd.Series

        :return: diastolicpeak location of dicrotic notches, 1-d array
        """

        nan_v = np.empty(len(dicroticnotch))
        nan_v[:] = np.NaN
        diastolicpeak = nan_v
        vpg = np.gradient(ppg)

        start_diag = diag_pulses[::2]
        # end_diag = diag_pulses[1::2]
        # end_diag = diag_pulses[2::2]
        end_diag = start_diag + 100

        for i in range(0,len(dicroticnotch)):
            try:
                len_segments=(end_diag - start_diag)*0.80
                end_segment=int(start_diag[i]+len_segments[i])
                try:
                    start_segment = int(dicroticnotch[i])
                    temp_segment = ppg[start_segment:end_segment]
                    max_locs, _ = find_peaks(temp_segment)

                    if len(max_locs)==0:
                        start_segment = int(e_point[i])
                        temp_segment = vpg[start_segment:end_segment]
                        max_locs, _ = find_peaks(temp_segment)

                except:
                    pass

                max_dn = max_locs[0] + start_segment
                diastolicpeak[i] = max_dn
            except:
                pass

        return diastolicpeak

def get_vpg_fiducials(ppg: np.array, onsets: list):
        """Calculate first derivitive points u and v from the PPG' signal

        :param onsets: onsets of the signal
        :type onsets: list

        :return:
            - u: The highest amplitude between the pulse onset and systolic peak on PPG'
            - v: The lowest amplitude between the u-point and diastolic peak on PPG'
            - w: The first local maximum or inflection point after the dicrotic notch on PPG’
        """

        dx = np.gradient(ppg)

        nan_v = np.empty(len(onsets)-1)
        nan_v[:] = np.NaN
        u, v, w = copy.deepcopy(nan_v),copy.deepcopy(nan_v),copy.deepcopy(nan_v),

        for i in range(0,len(onsets)-1):
            try:
                segment = dx[onsets[i]:onsets[i + 1]]

                # u fiducial point
                max_loc = np.argmax(segment)+onsets[i]
                u[i]=max_loc

                # v fiducial point
                upper_bound_coeff = 0.66
                v_upper_bound = ((onsets[i + 1] - onsets[i]) * upper_bound_coeff + onsets[i]).astype(int)
                min_loc = np.argmin(dx[int(u[i]):v_upper_bound])+u[i]-1
                v[i] = min_loc

                # w fiducial point
                temp_segment=dx[int(v[i]):onsets[i+1]]
                max_locs, _ = find_peaks(temp_segment)
                max_w = max_locs[0] + v[i] - 1
                w[i] = max_w

            except:
                pass

        vpg_fp = pd.DataFrame({"u":[], "v":[], "w":[]})
        vpg_fp.u, vpg_fp.v, vpg_fp.w = u, v, w
        return vpg_fp