from __future__ import division
from collections import defaultdict
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import itertools

from .study import Study, MRMCStudy


class EFROC(Study):
    """
    Has information about an Exponential Transformation of the Free Response Reciever Operating Characteristic curve.
    This is a subclass of the abstract base class :class:`study` which is used for EFROC studies.
    For more on EFROC analysis, see L. M. Popescu, `Med. Phys.` 38:5690, 2011
    Nonparametric signal detectability evaluation using an exponential transformation of the FROC curve
    (http://dx.doi.org/10.1118/1.3633938)

    :ivar area_scanned: dictionary that maps the cases to the area scanned by them.
    :ivar area_scanned_total: the total area scanned
    :ivar reference_area: The reference area for the figure of merit.
    :ivar n: The number of images from where false-positive scores are taken.
    :ivar true_cases: dictionary that maps the cases to the true scores in them
    :ivar true: list of all the true scores
    :ivar false_cases: dictionary that maps the cases to the false scores in them
    :ivar false: list of all the false scores
    :ivar signal_count_cases: dictionary that maps all the cases to the signal counts in them
    :ivar signal_count: the total number of signals present
    :ivar estimation: The estimated area under the curve.
    :ivar variance: The variance of the estimated area under the curve.
    :ivar name: A name for the study object.
    """

    def _variance_calculator(self, h=None):
        """
        Calculate the variance of the Area under the Curve estimation

        Overrides method in abstract base class.

        :param h: function used to compare present and absent values. Defaults to study._heavi

        :return: variance of area under the EFROC curve
        """
        if h is None: h = self._heavi
        i = float(self.signal_count)  # cast to float so that division works correctly below
        self.s12 = self._s_one_two(self.true, self.false, self.n, h)
        self.s21 = self._s_two_one(self.true, self.false, self.n, h)
        self.s11 = (self._s_one_one(self.true, self.false, self.n, h))
        var = ((1 / i) * self.s12 + ((i - 1) / i) * self.s21 - self.s11 ** 2)
        return var

    def _auc_estimator(self, h=None):
        """
        Estimate the area under the EFROC curve.

        Overrides method in abstract base class

        :param h: function used to compare present and absent values. Defaults to study._heavi

        :return: estimated area under the EFROC curve.
        """
        if h is None: h = self._heavi
        present = np.array(self.true)
        absent = np.array(self.false)
        q = absent[:, None] - present[:, None].T  # be aware, this broadcasts in a different way then matlab.
        exp_val = np.sum(h(q), 0)
        a = np.sum(np.exp(-1 / self.n * exp_val))
        return a / self.signal_count

    def __init__(self, area_scanned, reference_area, true, false, signal_count, h=None, calc_var=True, name=''):
        """
        Constructor for EFROC

        This constructor has lots of parameters. It is usually better to load from a file using parse_and_create or
        parse_and_create_mrmc
        :param area_scanned: A dictionary which maps cases to area scanned in those cases.
        :param reference_area: The reference area for which we want to express the results
        :param true: A dictionary which maps cases to the true scores reported in those cases.
        :param false: A dictionary which maps cases to the false-signal scores reported in those images.
        :param signal_count: A dictionary which maps cases to the count of signals present in those images.
        :param h: The function used for comparing the true signal and false signal scores. Defaults to study._heavi
        :param calc_var: Determines if the constructor will calculate the variance.
        :param name: A name to be assigned to the EFROC object.
        :return: the EFROC object created
        """
        self.area_scanned = area_scanned
        self.area_scanned_total = sum(area_scanned.values())
        self.reference_area = reference_area
        self.n = float(self.area_scanned_total) / float(
            reference_area)  # making sure that n is a float for variance calc
        self.true_cases = true
        self.true = list(itertools.chain.from_iterable(true.values()))
        self.false_cases = false
        self.false = list(itertools.chain.from_iterable(false.values()))
        self.signal_count_cases = signal_count
        self.signal_count = sum(signal_count.values())
        self.estimation = self._auc_estimator(h)
        self.variance = self._variance_calculator(h) if calc_var else 0
        self.name = name

    def _s_one_one(self, true, false, n, h):
        """
        Helper function for variance.

        Uses nd-array to avoid looping in python.
        """
        c = n * (1 - math.exp(-1 / n))
        present = np.array(true)
        absent = np.array(false)
        q = absent[:, None] - present[:, None].T  # be aware, this broadcasts in a different way then matlab.
        exp_val = sum(h(q) * c, 0)
        a = np.sum(np.exp(-1 / self.n * exp_val))
        return a / self.signal_count

    def _s_one_two(self, true, false, n, h):
        """
        Helper function for variance.

        Uses nd-array to avoid looping in python.
        """
        c = n / 2 * (1 - math.exp(-2 / n))
        present = np.array(true)
        absent = np.array(false)
        q = absent[:, None] - present[:, None].T  # be aware, this broadcasts in a different way then matlab.
        exp_val = sum(h(q) * c, 0)
        a = np.sum(np.exp(-2 / self.n * exp_val))
        return a / self.signal_count

    def _s_two_one(self, true, false, n, h):
        """
        Helper function for variance.

        Uses nd-array to avoid looping in python.
        See Documentation/variance.tex for line-by-line description
        """
        c1 = n * (1 - math.exp(-1 / n))
        c3 = n * (math.exp(-1 / n) - math.exp(-2 / n))
        m = len(true)
        present = np.array(true)
        absent = np.array(false)
        q = absent[:, None] - present[:, None].T  # be aware, this broadcasts in a different way then matlab.
        nu = sum(h(q), 0)
        c1 = c1 * np.ones((m, m))
        c3 = c3 * np.ones((m, m))
        e1 = np.tril(c1, -1) + np.triu(c3)
        e2 = np.tril(c3, -1) + np.triu(c1)
        exp_val = -1.0 / n * (np.tile(nu[:, None], (1, m)) * e2 + nu * e1)  # cause it's row vs column
        s = np.sum(np.exp(exp_val))
        return s / (self.signal_count ** 2)

    def display_curve(self, figure=True):
        """
        Display the exponential transformation of the free response reciever operating characteristic curve.

        This is the curve that this class is generating information about

        :param figure: if the graph should be displayed in a new figure.

        :return: None
        """
        if figure: plt.figure()
        #thresh = np.linspace(-.1, end*1.1, 50000)
        thresh=np.sort(np.array(self.false))
        true_pos = np.zeros_like(thresh)
        false_neg = np.zeros_like(thresh)
        true_neg = np.zeros_like(thresh)
        false_pos = np.zeros_like(thresh)
        for true in self.true:
            true_pos[thresh < true] += 1
            false_neg[thresh >= true] += 1
        for false in self.false:
            false_pos[thresh < false] += 1
            true_neg[thresh >= false] += 1
        sensitivity = true_pos / (true_pos + false_neg)
        plt.plot(1 - np.exp(-false_pos / self.n), sensitivity, '-', label=self.name+"\nAUC: {0}".format(self.estimation))
        plt.legend(loc='lower right')
        plt.title("Exponential Transformation of FROC Curve")
        plt.xlabel("Projected fraction of false positive images")
        plt.ylabel("True Positive Rate (Sensitivity)")

    def __str__(self):
        """
        Write EFROC object to string.
        """
        string = "AFE={0}\n".format(self.estimation)
        string += "AFE_sig={0}\n\n".format(math.sqrt(self.variance))
        string += "S11={0}\n".format(self.s11)
        string += "S12={0}\n".format(self.s12)
        string += "S21={0}\n".format(self.s21)
        return string

    def resample(self, generator=random):
        """
        Resample with replacement to generate a new EFROC.

        This method is used in :class:`MRMC_EFROC` bootstrapping to estimate the variance
        of the difference of two modalities.

        :return: the EFROC object generated from the resampled cases.
        """
        resample_number = len(self.signal_count_cases)
        count_signals = {}
        false = {}
        true = {}
        area = {}
        for resampled_case in xrange(resample_number):
            case_select = generator.choice(self.true_cases.keys())
            false[resampled_case] = self.false_cases[case_select]
            true[resampled_case] = self.true_cases[case_select]
            area[resampled_case] = self.area_scanned[case_select]
            count_signals[resampled_case] = self.signal_count_cases[case_select]
        efroc = EFROC(area, self.reference_area, true, false, count_signals, calc_var=False)
        return efroc


class MRMCEFROC(MRMCStudy):
    """
    Compare the area under the curve of two modalities using EFROC.

    :ivar filename: Dictionary with the filenames of the two modalities being compared keyed with 1 and 2.
    :ivar modalities: Dictionary of dictionaries of :class:`EFROC`,
        associated with each reader's scores for each modality. The first dictionary is keyed with filenames, the second
        is keyed with reader id numbers.
    :ivar difference: The difference between the area under the curve of the two modalities.
    :ivar variance: The estimated variance of the difference between the two modalities. Set by :func:`bootstrap`
    """

    def __init__(self, keys, modalities, bootstrap=0, name=None, **kwargs):
        super(MRMCEFROC, self).__init__(EFROC, keys, modalities, bootstrap, name, **kwargs)


    def _two_modalities_variance(self):
        """
        Not implemented. Use :func:`bootstrap` to set the variance instead.

        :return: 0.0
        """
        return None

    def _resample_and_compare(self, gen):
        """
        Resamples each modality and reader combination to generate an average difference.

        :return: the average difference
        """
        resampled = defaultdict(dict)
        modality_average = {}
        state = gen.getstate()
        for modality, value in self.modalities.iteritems():
            for reader, results in value.iteritems():
                # make sure the state gets set, so that the same cases are selected for each reader and modality.
                gen.setstate(state)
                resampled[modality][reader] = results.resample(gen)
            modality_average[modality] = self.calculate_reader_auc_average(resampled[modality])
        return float(modality_average[self.keys[1]] - modality_average[self.keys[0]])

    def bootstrap(self, bootstrap_sample_size=1, seed=None):
        """
        Use bootstrapping to calculate the variance of the difference of two EFROC studies.

        :param bootstrap_sample_size: Number of times to resample. Defaults to 1.
        :param seed: Seed to initially pass to the random number generator. Defaults to None.

        :return: an ND Array of the bootstrapped differences.
        """
        difference_list = []
        gen = random.Random()
        gen.seed(seed)
        for count in xrange(bootstrap_sample_size):
            difference_list.append(self._resample_and_compare(gen))
            if seed is not None:
                random.jumpahead(seed)
        difference_array = np.array(difference_list)
        self.variance = np.var(difference_array)

        plt.figure()
        plt.hist(difference_array, np.ceil(np.sqrt(bootstrap_sample_size)), histtype='stepfilled')
        plt.title("Bootstrapped Estimation of $\delta A_{FE}$")
        plt.xlabel("$\delta A_{FE}$")
        plt.ylabel("Count")
        return difference_array
