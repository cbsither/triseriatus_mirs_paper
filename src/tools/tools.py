import numpy as np
import emd
from math import factorial
from numpy.linalg import inv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, f1_score

from sklearn.utils import resample

def sort_dataframe_by_indices(df, index_order):
    # Reindex the DataFrame based on the provided index order
    sorted_df = df.reindex(index_order)
    return sorted_df


def drop_rows_by_index(df, indices):
    """
    Drop rows from a DataFrame based on their index.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to drop rows.
    indices (list): A list of indices of the rows to be dropped.

    Returns:
    pandas.DataFrame: A new DataFrame with the specified rows dropped.
    """
    return df.drop(indices, axis=0)

def hdf_reader(file_path):
    store = pd.HDFStore(file_path)
    df = store.get('df')
    mdf = store.get('mdf')
    store.close()
    return mdf, df

def horizontal_reflection(arr):
    """Flips a numpy array or python list."""

    if isinstance(arr, list):
        arr.reverse()  # reverse the list in place
        return arr
    elif isinstance(arr, np.ndarray):
        return np.flip(arr, 1)  # flip the numpy array horizontally
    else:
        raise TypeError('arr must be np.ndarray or python list.')


def orientation_detector(arr):
    """
    Checks if the array is monotonically increasing.
    """
    diff = np.diff(arr)
    return np.all(diff > 0)


def sort_dataframe_by_indices(df, index_order):
    """
    Reindex the DataFrame based on the provided index order
    """
    sorted_df = df.copy().reindex(index_order)
    return sorted_df


def label_encoder(input_list):
    """
    Takes a python list and transforms it into numbers
    """
    # create a unique mapping for the values
    unique_values = list(set(input_list))
    value_to_number = {value: idx for idx, value in enumerate(unique_values)}

    # transform the original list based on the mapping
    transformed_list = [value_to_number[value] for value in input_list]

    return transformed_list, value_to_number


def mlecauchy(x, toler=0.001):
    """
    From: https://stats.stackexchange.com/questions/174117/maximum-likelihood-estimator-of-location-parameter-of-cauchy-distribution

    :param x: a vector of samples from a cauchy distribution
    :param toler: minimum tolerance
    :return: theta estimate
    """
    x = np.asarray(x)  # Ensure x is a NumPy array
    startvalue = np.median(x)
    n = len(x)
    thetahatcurr = startvalue

    # Compute first derivative of log likelihood
    firstderivll = 2 * np.sum((x - thetahatcurr) / (1 + (x - thetahatcurr) ** 2))

    # Continue Newton’s method until the first derivative is within tolerance
    while abs(firstderivll) > toler:
        # Compute second derivative of log likelihood
        secondderivll = 2 * np.sum(((x - thetahatcurr) ** 2 - 1) / (1 + (x - thetahatcurr) ** 2) ** 2)

        # Newton’s method update of estimate of theta
        thetahatnew = thetahatcurr - firstderivll / secondderivll
        thetahatcurr = thetahatnew

        # Compute first derivative of log likelihood
        firstderivll = 2 * np.sum((x - thetahatcurr) / (1 + (x - thetahatcurr) ** 2))

    return {'thetahat': thetahatcurr}


class Spectra:

    def __init__(self, x, y):
        self.spectra = None




class Evaluation:

    def __init__(self):
        pass

    def evaluation_metrics_table(self, y_pred, y_actual, n_bootstraps=1000):

        # Calculate metrics
        recall = recall_score(y_actual, y_pred)
        precision = precision_score(y_actual, y_pred)
        accuracy = accuracy_score(y_actual, y_pred)
        f1 = f1_score(y_actual, y_pred)

        # Create arrays to store bootstrap results
        recall_scores = []
        precision_scores = []
        accuracy_scores = []
        f1_scores = []

        # Perform bootstrapping
        for _ in range(n_bootstraps):
            # Resample the data
            indices = resample(range(len(y_actual)), replace=True)
            y_true_resampled = y_actual.iloc[indices]
            y_pred_resampled = np.array(y_pred)[indices]

            # Compute metrics for resampled data
            recall_scores.append(recall_score(y_true_resampled, y_pred_resampled))
            precision_scores.append(precision_score(y_true_resampled, y_pred_resampled))
            accuracy_scores.append(accuracy_score(y_true_resampled, y_pred_resampled))
            f1_scores.append(f1_score(y_true_resampled, y_pred_resampled))

        # Calculate confidence intervals (e.g., 95%)
        recall_ci = np.percentile(recall_scores, [2.5, 97.5])
        precision_ci = np.percentile(precision_scores, [2.5, 97.5])
        accuracy_ci = np.percentile(accuracy_scores, [2.5, 97.5])
        f1_ci = np.percentile(f1_scores, [2.5, 97.5])

        # Print confidence intervals
        print(f"Recall CI: {recall_ci}")
        print(f"Precision CI: {precision_ci}")
        print(f"Accuracy CI: {accuracy_ci}")
        print(f"F1 Score CI: {f1_ci}")
        # Create a table for metrics
        metrics_table = pd.DataFrame({
            "Metric": ["Recall/Sensitivity", "Precision/Specificity", "Accuracy", "F1 Score"],
            "Value": [recall, precision, accuracy, f1],
            "CI (2.5%)": [recall_ci[0], precision_ci[0], accuracy_ci[0], f1_ci[0]],
            "CI (97.5%)": [recall_ci[1], precision_ci[1], accuracy_ci[1], f1_ci[1]]
        })

        return metrics_table




class Preprocess:

    def __init__(self):
        pass

    def findbn(self, xvec, wl):
        """finds the closest vector index in x given a wavelength in the same units as x
        returns a single scaler integer"""
        bn = np.round(np.abs(xvec - wl).argmin())
        return int(bn)


    def crop(self, vec, hi, lo):
        """
        crop data down to a smaller size
        newvec = crop(vec,hi,lo)
        """
        newvec = vec[lo:hi]
        return newvec


    def sg(self, y, window_size, polyorder, deriv=0, rate=1):
        """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        polyorder : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688

        NOTE:  adapted from SCIPY: cookbook
        """

        try:
            window_size = np.abs(np.int64(window_size))
            order = np.abs(np.int64(polyorder))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise (TypeError("window_size size must be a positive odd number"))
        if window_size < order + 2:
            raise (TypeError("window_size is too small for the polynomials order"))
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

    def sg_dataframe(self, df, windowsize, polyorder, deriv):
        """
        Applies a Savitzky-Golay filter (and derivative) row-wise
        to a DataFrame, returning a new DataFrame that retains the
        original index and as many original column names as fit
        the new data length.

        Parameters
        ----------
        df : pd.DataFrame
            Each row is a spectrum; columns are wavenumbers (or similar).
        windowsize : int
            SG window size.
        polyorder : int
            SG polynomial order.
        deriv : int
            Derivative order (0 = no derivative, 1 = first derivative, etc.).

        Returns
        -------
        Adf : pd.DataFrame
            The SG-transformed data, with the same index as `df`,
            and columns matching the original names (up to the length
            of the transformed spectrum).
        """
        # Apply SG to the first row to see how many points we get back
        first_spectrum = df.iloc[0].values
        first_transformed = self.sg(first_spectrum,
                                    window_size=windowsize,
                                    polyorder=polyorder,
                                    deriv=deriv)
        spec_length = len(first_transformed)

        # Retain the original index
        new_index = df.index

        # If the transformed spectrum is the same length, keep all column names
        if spec_length == df.shape[1]:
            new_columns = df.columns
        else:
            # Otherwise, keep the first 'spec_length' column names
            new_columns = df.columns[:spec_length]

        # Create an empty DataFrame with the same index and adjusted columns
        Adf = pd.DataFrame(index=new_index, columns=new_columns)

        # Fill the new DataFrame row-by-row
        for idx in df.index:
            original_spec = df.loc[idx].values
            transformed_spec = self.sg(original_spec,
                                       window_size=windowsize,
                                       polyorder=polyorder,
                                       deriv=deriv)
            # Assign to the new DataFrame
            Adf.loc[idx] = transformed_spec

        return Adf

    def hnorm(self, y, ntype, **blvar):
        """this function will normalize or autoscale the data

        usage:  hnorm(y,ntype,**blvar)
        where
        y is the data that you want to normalize
        ntype is the type of normalization
        ntype = 1 max band is set to 1
        ntype = 2 area under y is 1
        ntype = 3 specific band is set to 1
              blvar needs to be a dictionary
              blvar = {'x':x,'wl':wl}
              where x is the x vector that goes with y
              and wl is the wavelength that you want to set to one
        ntype = 4 is an autoscaling function

        y = (y-mean y)/STDy
        created Oct 29, 2014, swh
        last modified Oct 29, 2014"""

        if ntype == 1:
            tmp = y - np.min(y)
            newy = tmp / np.max(tmp)
        elif ntype == 2:
            tmp = y - np.min(y)
            newy = tmp / np.linalg.norm(tmp)
        elif ntype == 3:
            if len(blvar) > 0:
                wl = blvar['wl']
                x = blvar['x']
            bn = self.findbn(x, wl)
            tmp = y - np.min(y)
            newy = tmp / tmp[bn]
        elif ntype == 4:
            std = np.std(y)
            m = np.mean(y)
            print(std, m)
            newy = (y - m) / std
        else:
            newy = y
        return newy

    def hnorm_df(self, df, ntype):

        # Initialize an empty DataFrame with the correct number of columns
        spec_length = len(self.hnorm(y=df.iloc[0].values, ntype=ntype))
        Adf = pd.DataFrame(index=df.index, columns=df.columns)

        # Process each spectrum in the DataFrame
        for index in df.index:
            spec = df.loc[index].values  # Extract spectrum for the given index
            spec = self.hnorm(y=spec, ntype=ntype)  ## norm

            # Add the processed spectrum as a new row in the DataFrame
            Adf.loc[index] = spec

        return Adf

    def saopca(self, A, nlv):
        '''
        successive average orthogonalization determination of scores and loading vectors
        usage: s,L = saopca(A,nlv)
        '''
        nspec, npts = A.shape
        maxnumlv = nlv
        scores = np.zeros((nspec, nlv))
        loadvec = np.zeros((npts, nlv))
        Ravg = np.zeros((1, npts))
        s = np.matrix(np.ones((nspec, 1)))
        Eigenold = 10000000
        err = 0.00000000001  # 1e-11
        for k in range(maxnumlv):
            deside = 1
            while np.abs(deside) > err:
                v = ((inv(s.T * s)) * s.T * A).T  ## this A is in (numspec,numpts)
                # v = v/np.sqrt(np.dot(v,v))
                v = v / np.sqrt(np.dot(np.squeeze(np.array(v)), np.squeeze(np.array(v.T))))  ## normalize
                s = A * v
                Eigen = np.dot(np.squeeze(np.array(s)), np.squeeze(np.array(s)))
                deside = Eigenold - Eigen
                Eigenold = Eigen

            R = A - s * v.T
            A = R
            scores[:, k] = np.squeeze(np.array(s[:, 0]))
            loadvec[:, k] = np.squeeze(np.array(v[:, 0]))
            ##setting up next score
            Ravg[0, :] = A[0, :]
            for j in range(nspec):
                if np.dot(np.squeeze(np.array(Ravg)), np.squeeze(np.array(A[j, :]))) < 0:
                    s[j, 0] = -1
                else:
                    s[j, 0] = 1

                Ravg = Ravg + s[j, 0] * A[j, :]

        loadvec = loadvec.T
        return scores, loadvec

    def rs_pf(self, x, y, degree):
        """
        Calculates the R-squared coefficient for polynomial fitting.

        Args:
            x (array-like): Independent variable values.
            y (array-like): Dependent variable values.
            degree (int): Degree of the polynomial fit.

        Returns:
            float: R-squared value.
        """
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        yhat = p(x)  # Predicted values
        ybar = np.mean(y)  # Mean of y
        ssreg = np.sum((yhat - ybar) ** 2)  # Explained sum of squares
        sstot = np.sum((y - ybar) ** 2)  # Total sum of squares
        return ssreg / sstot

    # Function to discard spectra based on polynomial fitting
    def discard_spectra_with_bad_fitting(self, df, rs_thres=0.99):
        """
        Discards spectra from a pandas DataFrame if they poorly fit a 5th-degree polynomial
        in the 3900-3500 cm^-1 range.

        Args:
            df (pd.DataFrame): The DataFrame containing spectra. Index represents spectrum names,
                               columns are wavelengths.

        Returns:
            pd.DataFrame: A filtered DataFrame with spectra that fit poorly removed.
            list: List of discarded spectrum names.
        """
        bad_spectra = []
        spectra_to_keep = []
        bs = 0  # Counter for discarded spectra

        wavelengths = df.columns

        for spectrum_name, spectrum in df.iterrows():
            # Identify the indices corresponding to the wavelength range 3500-3900
            mask = (wavelengths >= 3500) & (wavelengths <= 4000)
            relevant_wavelengths = wavelengths[mask]
            relevant_intensities = spectrum[mask]

            # If the spectrum doesn't reach the required range
            if len(relevant_intensities) == 0:
                raise Exception(f"The spectrum {spectrum_name} doesn't reach 3900 cm-1")

            # Generate x (relative indices) and y (intensities)
            x = np.arange(len(relevant_intensities))
            y = relevant_intensities.values

            # Calculate R-squared for 5th-degree polynomial fitting
            rs = self.rs_pf(x, y, 5)

            # Discard spectrum if R-squared is below the threshold
            if rs < rs_thres:
                bs += 1
                bad_spectra.append(f"AI: {spectrum_name}")
            else:
                spectra_to_keep.append(spectrum_name)

        # Filter the DataFrame to remove bad spectra
        filtered_df = df.loc[spectra_to_keep]

        # Print summary
        if bs == 1:
            print("1 spectrum has been discarded because it has atmospheric interferences.")
        else:
            print(f"{bs} spectra have been discarded because they have atmospheric interferences.")

        return filtered_df, bad_spectra

    def discard_low_intensity_spectra(self, df, mu_thres=0.11):
        """
        Discards spectra from a pandas DataFrame based on low intensity in a specified wavelength range.

        Args:
            df (pd.DataFrame): The DataFrame containing spectra. Index represents spectrum names,
                               columns are wavelengths.

        Returns:
            pd.DataFrame: A filtered DataFrame with low-intensity spectra removed.
            list: List of discarded spectrum names.
        """
        bad_spectra = []
        spectra_to_keep = []

        for spectrum_name, spectrum in df.iterrows():
            # Determine the wavelength range to evaluate
            wavelengths = df.columns
            intensity_values = spectrum.values

            # Identify the indices corresponding to the wavelength range 400-600
            mask = (wavelengths >= 400) & (wavelengths <= 800)
            relevant_intensities = intensity_values[mask]

            # If the spectrum doesn't reach 600 cm-1
            if len(relevant_intensities) == 0:
                # raise Exception(f"The spectrum {spectrum_name} doesn't reach 600 cm-1")
                print(f"The spectrum {spectrum_name} doesn't reach 600 cm-1")

            # Check if the average intensity in this range is above the threshold
            if np.mean(relevant_intensities) < mu_thres:
                bad_spectra.append(spectrum_name)
            else:
                spectra_to_keep.append(spectrum_name)

        # Filter the DataFrame to remove bad spectra
        filtered_df = df.loc[spectra_to_keep]

        # Print summary
        if len(bad_spectra) == 1:
            print("1 spectrum has been discarded because of its low intensity.")
        else:
            print(f"{len(bad_spectra)} spectra have been discarded because of their low intensity.")

        return filtered_df, bad_spectra

    def discard_distorted_spectra(self, df, l=1.5):
        """
        Discards spectra from a pandas DataFrame based on outlier detection at 1900 cm^-1.

        Args:
            df (pd.DataFrame): The DataFrame containing spectra. Index represents spectrum names,
                               columns are wavelengths.
            l (float): Multiplier for the interquartile range to define outlier fences.

        Returns:
            pd.DataFrame: A filtered DataFrame with distorted spectra removed.
            list: List of discarded spectrum names.
        """
        bad_spectra = []
        spectra_to_keep = []
        bs = 0  # Counter for discarded spectra

        # Check if 1900 cm^-1 is within the DataFrame's columns
        if 1900 not in df.columns:
            raise Exception("The wavelength 1900 cm^-1 is not present in the DataFrame.")

        # Extract intensities at 1900 cm^-1
        li = df[1900].values

        # Calculate the interquartile range and fences
        q3, q1 = np.percentile(li, [75, 25])
        ir = q3 - q1
        upper_fence = q3 + l * ir
        lower_fence = q1 - l * ir

        # Identify and discard outliers
        for spectrum_name, intensity in df[1900].items():
            if intensity > upper_fence or intensity < lower_fence:
                bs += 1
                bad_spectra.append(f"SA: {spectrum_name}")
            else:
                spectra_to_keep.append(spectrum_name)

        # Filter the DataFrame to remove bad spectra
        filtered_df = df.loc[spectra_to_keep]

        # Print summary
        if bs == 1:
            print("1 spectrum has been discarded because it was distorted by the anvil.")
        else:
            print(f"{bs} spectra have been discarded because they were distorted by the anvil.")

        return filtered_df, bad_spectra


    def emd_filter(self, y, imfs, max_imfs):

        imf = emd.sift.sift(y, max_imfs)

        if abs(imfs[0] - imfs[1]) >= 1:
            return imf[:,imfs[0]:imfs[1]].sum(axis=1)
        else:
            return imf[:,imfs[0]]


    def emd_filter_df(self, df, imfs, max_imfs):

        # Initialize an empty DataFrame with the correct number of columns
        spec_length = len(df.iloc[0].values)
        Adf = pd.DataFrame(index=df.index, columns=df.columns)

        for index in df.index:
            spec = df.loc[index].values
            spec = self.emd_filter(y=spec, imfs=imfs, max_imfs=max_imfs)

            Adf.loc[index] = spec

        return Adf