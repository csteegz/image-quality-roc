from __future__ import division
from __future__ import print_function
import math

import numpy as np
import matplotlib.pyplot as plt
from .study import Study, MRMCStudy, perr


class LROC(Study):
    """
    Has information about the Localized Reciever Operating Characteristics Curve of a device.
    This is a subclass of the abstract base class :class:`Study` that is used for LROC studies.
    It overrides init, parse_and_create, parse_and_create_mrmc, _auc_estimator, _variance_calculator, and also extends
    it with the method parse_and_create_mm.

    :ivar present: list of scores from signal present images
    :ivar absent: list of scores from signal absent images
    :ivar success: list of correct localization flag for scores from signal present images
    :ivar localization_fraction: estimation of the correct localization rate
    :ivar localization_variance: estimated variance of the localization_fraction
    :ivar estimation: estimated area under the LROC curve.
    :ivar variance: estimated variance in estimation.
    :ivar name: Name of the case
    """
#    parse_and_create = file_parser.ParseAndCreateLROC()
#    parse_and_create_mm_column = file_parser.ParseAndCreateLROCMMColumn()
#    parse_and_create_mm_label = file_parser.ParseAndCreateLROCMMLabel()
    # noinspection PyDefaultArgument,PyDefaultArgument
    def __init__(self, present=[], absent=[], success=[], k=None, calc_var=True, name=""):
        """
            Create LROC object

            In most cases, one should use the factories that create an LROC object from a file instead of this
            constructor.

            :param present: List of confidence scores associated with signal-present ROIs.
            :param absent: List of confidence scores associated with signal-absent ROIs.
            :param success: List of localization success scores for signal-present ROIs.
            :param k: comparison function. When None, methods default to using :py:func:`Study.Study._heavi`
            :param calc_var: allows disabling calculation of variance in order to speed up object instantiation.
            :param name: optional name parameter, for prettier graphing and maybe sorting.

            :return: an instance of LROC
            """
        self.present = present
        self.absent = absent
        self.success = success
        self.localization_fraction = float(success.count(1)) / len(present)
        self.localization_variance = (self.localization_fraction / (len(present) - 1)) * (
            1 - self.localization_fraction)
        self.estimation = self._auc_estimator(k)
        self.variance = self._variance_calculator(k) if calc_var else 0
        self.name = name

    def _auc_estimator(self, k=None):
        """
        Calculates and returns the index estimator using object variables

        Overrides method in abstract base class Sudy
        :param k: comparison function. Defaults to using :py:func:`Study.Study._heavi`
        :rtype : float
        :return: The performance index estimator for a list of confidence scores and localization
        success scores.
        """
        if k is None: k = self._heavi
        m = len(self.present)
        n = len(self.absent)
        present = np.array(self.present)
        absent = np.array(self.absent)
        success = np.array(self.success)
        y = present - np.transpose([absent])
        q = np.multiply(k(y), success)
        a = np.sum(q)
        a /= (m * n)
        return a

    def _variance_calculator(self, k=None):
        """
        Calculates and returns the variance of the index estimator using object variables

        Overrides method in abstract base class Study
        :param k: comparison function. Defaults to using :py:func:`Study.Study._heavi`
        :return: The variance of the performance index estimator.
        """
        m = float(len(self.present))
        n = float(len(self.absent))  # guarantee float division below
        if k is None: k = self._heavi
        a_squared = self.estimation ** 2
        # variance is fixed_reader_covariance with itself
        # call components of fixed_reader_covariance instead of the fixed_reader_covariance function so that they can be stored and output.
        self.mu11 = self._cov_beta_one_one(self.present, self.absent, self.success, self.present, self.absent,
                                           self.success, k) - a_squared
        self.mu12 = self._cov_beta_general(self.present, self.absent, self.success, self.present, self.absent,
                                           self.success, k,1) - a_squared
        self.mu21 = self._cov_beta_general(self.present, self.absent, self.success, self.present, self.absent,
                                           self.success, k,0) - a_squared
        return (self.mu11 + (n - 1) * self.mu12 + (m - 1) * self.mu21) / (m * n)

    @classmethod
    def covariance(cls, present_one, absent_one, success_one, present_two, absent_two, success_two, auc_one, auc_two,
                   k):
        """
        Calculate the fixed_reader_covariance between observations of one reader and observations of another reader, assuming the
        readers are fixed effects.

        :param present_one: Present observations of the first reader
        :param absent_one: absent observations of the first reader
        :param success_one: correct localizaton coefficients for the first reader
        :param present_two: present observations of the second reader
        :param absent_two: absent observations of the second reader
        :param success_two: correct localization coefficients for the second reader
        :param auc_one: area under the curve estimation based on first reader observations
        :param auc_two: area under the curve estimatino based on second reader observations
        :param k: comparison function. Defaults to using :py:func:`Study.Study._heavi`
        :return: the fixed_reader_covariance
        """
        m = len(present_one)
        n = len(absent_one)
        mu_one_one = cls._cov_beta_one_one(present_one, absent_one, success_one, present_two, absent_two, success_two,
                                           k) - auc_one * auc_two
        mu_two_one = cls._cov_beta_general(present_one, absent_one, success_one, present_two, absent_two, success_two,
                                           k,0) - auc_one * auc_two
        mu_one_two = cls._cov_beta_general(present_one, absent_one, success_one, present_two, absent_two, success_two,
                                           k,1) - auc_one * auc_two
        cov = (mu_one_one + (m - 1) * mu_two_one + (n - 1) * mu_one_two) / (m * n)
        return cov

    @classmethod
    def _cov_beta_one_one(cls, present_one, absent_one, success_one, present_two, absent_two, success_two, k):
        """
        Function which calculates a sub-part of the variance. Helper function for _variance_calculator

        :param present_one: A list of M confidence scores for signal present ROI's
        :param absent_one: A list of N confidence scores for signal absent ROI's
        :param success_one: A list of M localization success scores
        :rtype : float
        :param k: The function to compare the things with. Uses Heaviside function.
        :return: A portion of the variance.
        """
        m = len(present_one)
        n = len(absent_one)
        present_one = np.array(present_one)
        absent_one = np.array(absent_one)
        success_one = np.array(success_one)
        present_two = np.array(present_two)
        absent_two = np.array(absent_two)
        success_two = np.array(success_two)
        # expand to two dimensions, M by N
        y_one = present_one - np.transpose([absent_one])
        y_two = present_two - np.transpose([absent_two])
        # still two dimensions
        q_one = k(y_one) * success_one
        q_two = k(y_two) * success_two
        # element by element multiplication
        q = q_one * q_two
        b = np.sum(q)
        b /= (m * n)
        return b

    @classmethod
    def _cov_beta_general(cls, present_one, absent_one, success_one, present_two, absent_two, success_two, k, dim):
        """
        Function which calculates a sub-part of the fixed_reader_covariance. Helper function for _variance_calculator

        :param present_one: A list of M confidence scores for signal present ROI's
        :param absent_one: A list of N confidence scores for signal absent ROI's
        :param success_one: A list of M localization success scores
        :rtype : float
        :param k: The function to compare the things with. uses heaviside function
        :return: A portion of the variance.
        """
        present_one = np.array(present_one)
        absent_one = np.array(absent_one)
        success_one = np.array(success_one)
        present_two = np.array(present_two)
        absent_two = np.array(absent_two)
        success_two = np.array(success_two)
        # expand to m by n. m will be dimension 0, n is dimension 1. dim is the dimension we are iterating over twice.
        y_one = np.transpose([present_one]) - absent_one
        y_two = np.transpose([present_two]) - absent_two
        # apply successes to correct places
        q_one = k(y_one) * success_one[:,None]
        q_two = k(y_two) * success_two[:,None]
        # expand to m by n by dim
        transpose_matrix = [0,1,2]
        transpose_matrix[dim],transpose_matrix[2]=transpose_matrix[2],transpose_matrix[dim]
        q = q_one[:, :, None] * np.transpose(q_two[:, :, None], transpose_matrix)
        divisor = q.shape[0] * q.shape[1] * (q.shape[2]-1)
        # replace the main diagonal with zeroes.
        #the 1-dim is a hack to get the non-passed dimension
        q *= np.expand_dims(np.logical_not(np.eye((q.shape[dim]))),1-dim)
        b = np.sum(q)
        b /= divisor
        return b

    def display_curve(self, figure=True):
        """
        Calculate and display the localization Reciever Operating Characteristics Curve

        :param figure: A flag that specifies if the function should create a new figure. To plot multiple ROC's on the
            same axis, set to false.
        :return: The false positive rate and the correctly localized true positive rate
        :rtype: tuple of numpy.ndarray
        """
        #thresh = np.linspace(begin, end * 1.1, 50000)
        thresh = np.sort(np.array(self.absent))
        true_pos = np.zeros_like(thresh)
        false_neg = np.zeros_like(thresh)
        true_neg = np.zeros_like(thresh)
        false_pos = np.zeros_like(thresh)
        true_pos_local = np.zeros_like(thresh)
        for index in xrange(len(self.present)):
            true_pos[(thresh < self.present[index])] += 1
            true_pos_local[np.logical_and(thresh < self.present[index], self.success[index])] += 1
            false_neg[thresh >= self.present[index]] += 1
        for false in self.absent:
            false_pos[thresh < false] += 1
            true_neg[thresh >= false] += 1
        true_positive_localization_fraction = true_pos_local / (true_pos + false_neg)
        specificity = true_neg / (true_neg + false_pos)
        if figure: plt.figure()
        plt.plot(1 - specificity, true_positive_localization_fraction, '-', label=self.name+"\nAUC:{0}".format(self.estimation))
        plt.legend(loc='lower right')
        plt.title("Localized Receiver Operating Characteristic Curve")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("Correctly Localized True Positive Fraction")
        plt.axis([0, 1, 0, 1])
        return 1 - specificity, true_positive_localization_fraction

    def __str__(self):
        """
        Display information about the LROC study.

        :return: A string that displays information of the LROC study
        """
        string = "AL={0}\n".format(self.estimation)
        string += "AL_sig={0}\n\n".format(math.sqrt(self.variance))
        string += "QL={0}\n".format(self.localization_fraction)
        string += "QL_sig={0}\n\n".format(math.sqrt(self.localization_variance))
        string += "mu_11={0}\n".format(self.mu11)
        string += "mu_12={0}\n".format(self.mu12)
        string += "mu_21={0}\n".format(self.mu21)
        return string


class ROC(LROC):
    """
    A subclass of LROC that is used to evaluate ROC studies.
    ROC studies can be represented as the specific case of an ROC study where all signals are correctly localized
    This would mean that every signal is a success.
    """
#    parse_and_create = file_parser.ParseAndCreateROC()
#    parse_and_create_mm_column = parser.ParseAndCreateROCMMColumn()
#    parse_and_create_mm_label = parser.ParseAndCreateROCMMLabel()
    def __init__(self, present=[], absent=[], success=[], k=None, calc_var=True, name=""):
        self.present = present
        self.absent = absent
        self.success = [1 in range(len(present))]
        self.estimation = self._auc_estimator(k)
        self.variance = self._variance_calculator(k) if calc_var else 0
        self.name = name

    def __str__(self):
        """
        Display information about the ROC study.

        :return: A string that displays information of the LROC study
        """
        string = "A={0}\n".format(self.estimation)
        string += "A_sig={0}\n\n".format(math.sqrt(self.variance))
        string += "mu_11={0}\n".format(self.mu11)
        string += "mu_12={0}\n".format(self.mu12)
        string += "mu_21={0}\n".format(self.mu21)
        return string

    def display_curve(self, figure=True):
        """
        Calculate and display the Reciever Operating Characteristics Curve

        :param figure: A flag that specifies if the function should create a new figure. To plot multiple ROC's on the
            same axis, set to false.
        :return: The false positive rate and the correctly localized true positive rate
        :rtype: tuple of numpy.ndarray
        """
        thresh = np.sort(np.array(self.absent))
        true_pos = np.zeros_like(thresh)
        false_neg = np.zeros_like(thresh)
        true_neg = np.zeros_like(thresh)
        false_pos = np.zeros_like(thresh)
        for index in xrange(len(self.present)):
            true_pos[(thresh < self.present[index])] += 1
            false_neg[thresh >= self.present[index]] += 1
        for false in self.absent:
            false_pos[thresh < false] += 1
            true_neg[thresh >= false] += 1
        true_positive_fraction = true_pos / (true_pos + false_neg)
        specificity = true_neg / (true_neg + false_pos)
        if figure:
            plt.figure()
            plt.plot([0,1],[0,1],linestyle='--',label="Pure Chance\nAUC:0.5")
        plt.plot(1 - specificity, true_positive_fraction, '-', label=self.name+"\nAUC:{0}".format(self.estimation))
        plt.legend(loc='lower right')
        plt.title("Receiver Operating Characteristic Curve")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Fraction")
        plt.axis([0, 1, 0, 1])
        return 1 - specificity, true_positive_fraction


class MRMCLROC(MRMCStudy):
    """
    Compare the area under the curve of two modalities using LROC.
    For more information see A. Wunderlich and F. Noo, `IEEE Trans. Med. Imag.`
    A Nonparametric Procedure for Comparing the Areas Under Correlated LROC Curves
    (http://dx.doi.org/10.1109/TMI.2012.2205015)

    :ivar filename: Dictionary with the filenames of the two modalities being compared keyed with 1 and 2.
    :ivar modalities: Dictionary of dictionaries of :class:`LROC`,
        associated with each reader's scores for each modality. The first dictionary is keyed with filenames,
        while the dictionary it contains is keyed with reader id numbers.
    :ivar difference: The difference between the area under the curve of the two modalities.
    :ivar variance: The estimated variance of the difference between the two modalities.
    """

    @staticmethod
    def fixed_reader_covariance(lroc_one,lroc_two):
        """
        Calculate the fixed_reader_covariance between two sets of observations, assuming the readers are fixed.

        :param lroc_one: The first set of observations to be compared
        :param lroc_two: The second set of observations to be compared

        :return: the fixed_reader_covariance between the two sets of observations
        """
        return lroc_one.covariance(lroc_one.present, lroc_one.absent, lroc_one.success, lroc_two.present,
                                   lroc_two.absent, lroc_two.success, lroc_one.estimation,
                                   lroc_two.estimation, lroc_one._heavi)

    def _two_modalities_variance(self):
        """
        Calculate the variance of the difference of two modalities.
        Assuming that the readers are fixed.

        :return: The variance of the difference of two modalities.
        """
        var = 0
        modality_one = self.modalities[self.keys[0]]
        modality_two = self.modalities[self.keys[1]]
        for key in self.modalities:
            for readers in self.modalities[key]:
                for more_readers in self.modalities[key]:
                    var += self.fixed_reader_covariance(self.modalities[key][readers],self.modalities[key][more_readers])
        for readers_one in modality_one:
            for readers_two in modality_two:
                var -= 2 * self.fixed_reader_covariance(modality_one[readers_one],modality_two[readers_two])
        return var


    def random_reader_variance(self):
        """
        O(R^2M^2N^2), so try really hard not to use unless necessary.
        Also might use a lot of memory.
        Still, profiles fairly well.
        Finally, if you haven't been scared off, I have no idea if this is correct.
        :return:
        """
        present_one = []
        present_two = []
        absent_one = []
        absent_two = []
        success_one = []
        success_two = []
        for reader in self.modalities[self.keys[0]]:
            present_one.append(self.modalities[self.keys[0]][reader].present)
            present_two.append(self.modalities[self.keys[1]][reader].present)
            absent_one.append(self.modalities[self.keys[0]][reader].absent)
            absent_two.append(self.modalities[self.keys[1]][reader].absent)
            success_one.append(self.modalities[self.keys[0]][reader].success)
            success_two.append(self.modalities[self.keys[1]][reader].success)
        (auc_one,auc_two) = self.average_area
        present_one = np.array(present_one).T
        absent_one = np.array(absent_one).T
        success_one = np.array(success_one).T
        present_two = np.array(present_two).T
        absent_two = np.array(absent_two).T
        success_two = np.array(success_two).T
        kwargs = {'present_one':present_one,'present_two':present_two,'absent_one':absent_one,
                  'absent_two':absent_two,'success_one':success_one,'success_two':success_two,'k':LROC._heavi,
                  'auc_one':auc_one,'auc_two':auc_two}
        return self._random_reader_covariance(**kwargs)

    def __init__(self, keys, modalities, bootstrap=0, name=None,random_readers=False,**kwargs):
        super(MRMCLROC, self).__init__(LROC, keys, modalities, bootstrap, name, **kwargs)
        self.covariance_components = {}
        self.success_rate_difference = self.success_diff()
        if random_readers:
            self.variance = self.random_reader_variance()
            self.success_rate_variance = self.random_reader_success_variance()
        else:
            self.success_rate_variance = self.fixed_reader_success_variance()



    def _random_reader_covariance(self,present_one, absent_one, success_one, present_two, absent_two, success_two, auc_one, auc_two,
                   k):
        kwargs = {'present_one':present_one,'present_two':present_two,'absent_one':absent_one,
                  'absent_two':absent_two,'success_one':success_one,'success_two':success_two,'k':k}
        (m , readers) = present_one.shape
        n = absent_one.shape[0]
        mu_111 = self._cov_one_one_one(**kwargs) - auc_one * auc_two
        mu_211 = self._cov_one_dim_general(dim=0,**kwargs) - auc_one * auc_two
        mu_121 = self._cov_one_dim_general(dim=1,**kwargs) - auc_one * auc_two
        mu_112 = self._cov_one_dim_general(dim=2,**kwargs)- auc_one * auc_two
        mu_221 = self._cov_two_dim_general(dims=(0,1),**kwargs) - auc_one * auc_two
        mu_212 = self._cov_two_dim_general(dims=(0,2),**kwargs) - auc_one * auc_two
        mu_122 = self._cov_two_dim_general(dims=(1,2),**kwargs) - auc_one * auc_two
        self.covariance_components = {'mu_111': mu_111,'mu_211':mu_211,'mu_121':mu_121,'mu_221':mu_221, 'mu_112':mu_112, 'mu_212':mu_212, 'mu_122':mu_122}
        b = (mu_111 + (m-1)*mu_211 + (n-1)
                                     * mu_121 + (m-1)*(n-1)*mu_221+(readers-1)*(mu_112 + (m-1) * mu_212 + (n-1)*mu_122))
        return b/(readers*m*n)

    @classmethod
    def _cov_one_one_one(cls,present_one, absent_one, success_one, present_two, absent_two, success_two,k):
        #must all be m by r or n by r arrays.
        present_one = np.array(present_one)
        absent_one = np.array(absent_one)
        success_one = np.array(success_one)
        present_two = np.array(present_two)
        absent_two = np.array(absent_two)
        success_two = np.array(success_two)
        (m,r) = present_one.shape
        n = absent_one.shape[0]
        # expand to three dimensions, M by N by R
        y_one = np.expand_dims(present_one,1) - np.expand_dims(absent_one,0)
        y_two = np.expand_dims(present_two,1) - np.expand_dims(absent_two,0)
        # still three dimensions
        q_one = k(y_one) * success_one[:,None,None]
        q_two = k(y_two) * success_two[:,None,None]
        # element by element multiplication
        q = q_one * q_two
        b = np.sum(q)
        b /= (m * n * r)
        return b

    @classmethod
    def _cov_one_dim_general(cls,present_one, absent_one, success_one, present_two, absent_two, success_two, k, dim):
        """

        :param present_one: A list of MbyR confidence scores for signal present ROI's
        :param absent_one: A list of NbyR confidence scores for signal absent ROI's
        :param success_one: A list of MbyR localization success scores
        :rtype : float
        :param k: The function to compare the things with. uses heaviside function
        :return: A portion of the variance.
        """
        present_one = np.array(present_one)
        absent_one = np.array(absent_one)
        success_one = np.array(success_one)
        present_two = np.array(present_two)
        absent_two = np.array(absent_two)
        success_two = np.array(success_two)
        # expand to m by n by r. m will be dimension 0, n is dimension 1, r is dimension 2. dim is the dimension we are iterating over twice.
        y_one = np.expand_dims(present_one,1) - np.expand_dims(absent_one,0)
        y_two = np.expand_dims(present_two,1) - np.expand_dims(absent_two,0)
        # apply successes to correct places. Still 3 dimensions
        q_one = k(y_one) * np.expand_dims(success_one,1)
        q_two = k(y_two) * np.expand_dims(success_two,1)
        # expand to m by n by r by dim
        transpose_matrix = [0,1,2,3]
        transpose_matrix[dim],transpose_matrix[3]=transpose_matrix[3],transpose_matrix[dim]
        q = q_one[:,:,:,None] * np.transpose(q_two[:,:,:,None], transpose_matrix)
        divisor = q.shape[0] * q.shape[1] * q.shape[2] * (q.shape[3]-1)
        #adding along the two non-passed dimensions
        transpose_matrix.remove(3)
        transpose_matrix.remove(dim)
        q = np.sum(q,tuple(transpose_matrix))
        # replace the main diagonal with zeroes.
        q *= np.logical_not(np.eye((q.shape[0])))
        b = np.sum(q)
        b /= divisor
        return b

    @classmethod
    def _cov_two_dim_general(cls,present_one, absent_one, success_one, present_two, absent_two, success_two, k,dims):
        """

        :param present_one: A list of MbyR confidence scores for signal present ROI's
        :param absent_one: A list of NbyR confidence scores for signal absent ROI's
        :param success_one: A list of MbyR localization success scores
        :rtype : float
        :param k: The function to compare the things with. uses heaviside function
        :return: A portion of the variance.
        """
        #I think this all has something to do with tensor products
        dim_one= sorted(dims)[0]
        dim_two= sorted(dims)[1]
        present_one = np.array(present_one)
        absent_one = np.array(absent_one)
        success_one = np.array(success_one)
        present_two = np.array(present_two)
        absent_two = np.array(absent_two)
        success_two = np.array(success_two)
        # expand to m by n by r (where a is either m or n). m will be dimension 0, n is dimension 1, r is dimension 2.
        # dim is the dimension we are iterating over twice.
        #obviously this is a loop under the hood, but a c/fortran loop is much faster than a python one.
        y_one = np.expand_dims(present_one,1) - np.expand_dims(absent_one,0)
        y_two = np.expand_dims(present_two,1) - np.expand_dims(absent_two,0)
        # apply successes to correct places. Still 4 dimensions
        q_one = k(y_one) * np.expand_dims(success_one,1)
        q_two = k(y_two) * np.expand_dims(success_two,1)
        # translate to m by n by r by one by one
        # and one by one by
        # unused dimension will be the middle, basically create 3 possible transpose_matrixes for the three cases
        transpose_matrix = [0,1,2,3,4]
        transpose_matrix[dim_one],transpose_matrix[3]=transpose_matrix[3],transpose_matrix[dim_one]
        transpose_matrix[dim_two],transpose_matrix[4]=transpose_matrix[4],transpose_matrix[dim_two]
        q = q_one[:,:,:,None,None] * np.transpose(q_two[:,:,:,None,None], transpose_matrix)
        divisor = q.shape[0] * q.shape[1] * q.shape[2] * (q.shape[3]-1) *(q.shape[4]-1)
        #adding along the non-passed dimension
        transpose_matrix.remove(3)
        transpose_matrix.remove(4)
        transpose_matrix.remove(dim_two)
        transpose_matrix.remove(dim_one)
        q = np.sum(q,tuple(transpose_matrix))
        #at this point we should have q, which is an a x b x a x b matrix.
        #get this into the form of an a x a x b x b matrix to multiply by the two diagonals
        q=np.transpose(q,[0,2,1,3])
        #replace the main diagonal with zeroes.
        diag_one = np.logical_not(np.eye((q.shape[0])))
        diag_two = np.logical_not(np.eye(q.shape[2]))
        diag = diag_one[:,:,None,None] * np.transpose(diag_two[:,:,None,None],[2,3,0,1])
        q *= diag
        b = np.sum(q)
        b /= divisor
        return b

    def __str__(self): # pragma: no cover
        """
        Displays information about the difference between the two modalities being compared.

        :return: A string with information about the modalities
        """
        string = self.name + "\n"
        string += ("DEL_A={0}\n".format(self.difference))
        if self.variance is not None:
            string += ("Del_A_Var={0}\n".format(self.variance))
        if self.success_rate_difference is not None:
            string += ("DEL_Q={0}\n".format(self.success_rate_difference))
            string += ("Del_Q_Var={0}\n".format(self.success_rate_variance))
        for key,value in self.covariance_components.iteritems():
            string += ("{0}={1}\n".format(key,value))
        string += "Specific Modalities\n\n"
        for key, value in self.modalities.iteritems():
            string += ("Name: {0}\nReader Count:{1}\n".format(key, len(value)))
            string += ("Average_Area={0}\nAverage_Area_Variance={1}\n".format(self.calculate_reader_auc_average(value),
                                                                            self.calculate_reader_var_average(value)))
            for reader, result in value.iteritems():
                string += "%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                string += ("Reader: {0}\n".format(reader))
                string += (str(result))
                string += "%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                string += "\n"
        return string

    def random_reader_success_variance(self):
        success_one = []
        success_two = []
        for reader in self.modalities[self.keys[0]]:
            success_one.append(self.modalities[self.keys[0]][reader].success)
            success_two.append(self.modalities[self.keys[1]][reader].success)
        theta_one = self.calculate_reader_succ_rate_average(self.modalities[self.keys[0]])
        theta_two = self.calculate_reader_succ_rate_average(self.modalities[self.keys[1]])
        success_one = np.array(success_one).T
        success_two = np.array(success_two).T
        m = success_one.shape[0]
        r = success_one.shape[1]
        mu_11 = np.sum(success_two * success_one) / success_one.size - theta_one * theta_two
        mu_21 = self._success_cov_one_dim_general(success_one,success_two,0) - theta_two * theta_one
        mu_12 = self._success_cov_one_dim_general(success_one,success_two,1) - theta_two * theta_one
        cov = mu_11 + (m-1) * mu_21 + (r-1)*mu_12
        cov /= (r*m)
        return cov


    @classmethod
    def _success_cov_one_dim_general(cls,success_one,success_two,dim):
        """

        :param present_one: A list of MbyR confidence scores for signal present ROI's
        :param absent_one: A list of NbyR confidence scores for signal absent ROI's
        :param success_one: A list of MbyR localization success scores
        :rtype : float
        :param k: The function to compare the things with. uses heaviside function
        :return: A portion of the variance.
        """
        success_one = np.array(success_one)
        success_two = np.array(success_two)
        # expand to m by r by dim
        transpose_matrix = [0,1,2]
        transpose_matrix[dim],transpose_matrix[2]=transpose_matrix[2],transpose_matrix[dim]
        q = success_one[:,:,None] * np.transpose(success_two[:,:,None], transpose_matrix)
        divisor = q.shape[0] * q.shape[1] * (q.shape[2]-1)
        #adding along the non-passed dimension
        transpose_matrix.remove(2)
        transpose_matrix.remove(dim)
        q = np.sum(q,tuple(transpose_matrix))
        # replace the main diagonal with zeroes.
        q *= np.logical_not(np.eye((q.shape[0])))
        b = np.sum(q)
        b /= divisor
        return b

    def fixed_reader_success_variance(self):
        var = 0
        modality_one = self.modalities[self.keys[0]]
        modality_two = self.modalities[self.keys[1]]
        for key in self.modalities:
            for readers in self.modalities[key]:
                for more_readers in self.modalities[key]:
                    var += self.fixed_reader_success_covariance(self.modalities[key][readers],self.modalities[key][more_readers])
        for readers_one in modality_one:
            for readers_two in modality_two:
                var -= 2 * self.fixed_reader_success_covariance(modality_one[readers_one],modality_two[readers_two])
        return var

    @staticmethod
    def fixed_reader_success_covariance(study_one,study_two):
        success_one = study_one.success
        success_two = study_two.success
        success_one = np.array(success_one)
        success_two = np.array(success_two)
        B = success_one[:,None] * np.transpose(success_two[:,None])
        B = np.sum(B) / len(success_one)
        cov = B - (study_one.localization_fraction*study_two.localization_fraction)
        cov /= len(success_one)
        return cov

    def success_diff(self):
        modality_average = {}
        for key,values in self.modalities.iteritems():
            modality_average[key] = self.calculate_reader_succ_rate_average(values)
        return modality_average[self.keys[1]] - modality_average[self.keys[0]]

    @staticmethod
    def calculate_reader_succ_rate_average(readers):
        total = 0.0
        for study in readers.itervalues():
            total += study.localization_fraction
        return total/len(readers)


class MRMCROC(MRMCLROC):

    def __init__(self, keys, modalities, bootstrap=0, name=None,random_readers=False,**kwargs):
        super(MRMCLROC, self).__init__(LROC, keys, modalities, bootstrap, name, **kwargs)
        self.covariance_components = {}
        self.success_rate_difference = None
        if random_readers:
            self.variance = self.random_reader_variance()
            self.success_rate_variance = None
        else:
            self.success_rate_variance = None
