from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import random
import sys
import numpy as np
import matplotlib.pyplot as plt


def perr(errorstring):
    print(errorstring, file=sys.stderr)


class Study:
    """
        Abstract Base Class that represents a Reciever Operating Characteristic Study for one observer.

        :ivar estimation: an estimate of the area under the curve.
        :ivar variance: an estimation of the variance of the area under the curve.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.estimation = 0.0
        self.variance = 0.0

    @abstractmethod
    def _auc_estimator(self):
        pass

    @abstractmethod
    def _variance_calculator(self):
        pass

    @staticmethod
    def _heavi(y):
        """ Evaluate the Heaviside function with H(0) = 1/2

        :param y: input to the heaviside function
        :return:  heaviside output
        """
        s = .5 * (y == 0) + (y > 0)
        return s

    def print_to_file(self, out=None):
        """
        Print information about the performance.

        :param out: The Prefix of the filename to be printed to. The file is 'out.auc'. If None, prints to 'tmp.auc'.

        :return: None
        """
        if out is None: out = 'tmp'
        out += '.auc'
        with open(out, 'w') as f:
            f.write(str(self))
        print("Output saved as: {0}".format(out))
        return None


class MRMCStudy:
    __metaclass__ = ABCMeta

    @staticmethod
    def calculate_reader_var_average(readers):
        total = 0.0
        for study in readers.itervalues():
            total += study.variance
        return total / len(readers)

    @staticmethod
    def calculate_reader_auc_average(readers):
        """
        Calculate the reader averaged area under the curve.

        :param readers: A dictionary of Study objects containing observations for each reader.

        :return: The reader-averaged area under the curve.
        """
        total = 0.0
        for study in readers.itervalues():
            total += study.estimation
        return total / len(readers)

    @staticmethod
    def calculate_reader_auc_var(readers):
        """
        Calculate the variance of the reader averaged area under the curve.

        :param readers: A dictionary of study objects containing observations for each reader.

        :return: The reader-averaged area under the curve.
        """
        val = np.array(readers.values())
        return np.var(val)

    def bootstrap_par(self, bootstrap_sample_size):
        """
        Use bootstrapping to calculate the variance of the difference of two EFROC studies.

        In testing, this is around 40% faster then non-threaded, but that will depend on the user's computer.
        Unfortunately, seeding can't be consistent between threaded and non-threaded, so the PRNG is always seeded with
        the timestamp.

        :param bootstrap_sample_size:
        :return: an ND array of the bootstrapped differences.
        """
        import multiprocessing

        pool = multiprocessing.Pool()
        # it's basically a map-reduce
        difference_list = pool.map(_Resampler(self.modalities, self.keys), range(bootstrap_sample_size))
        difference_array = np.array(difference_list)
        self.boot_variance = np.var(difference_array)
        return difference_array

    def compare_two_modalities(self):
        """
        Computes the difference between the AUC for the two modalities associated with the object.

        :return: the difference between the AUC for the two modalities associated with the object.
        """
        modality_average = {}
        for key, values in self.modalities.iteritems():
            modality_average[key] = self.calculate_reader_auc_average(values)
        return modality_average[self.keys[1]] - modality_average[self.keys[0]]

    def __init__(self, study_class, keys, modalities, bootstrap=0, name=None, **kwargs):
        """
        Instantiates an instance of MRMCStudy with two modalities and a type of :class:`Study` associated with it.

        Computes the difference of the AUC,and the variance of this difference. Creating an object of this type is the
        preferred way to compare two modalities using :class:`Study`.
        :param study_class:
        :param keys:
        :param modalities:
        :param bootstrap:
        :param name:
        :param exclude_readers:
        :return: the instance of MRMCStudy
        """
        self.keys = keys
        self.name = name if name is not None else ("Difference between {0} and {1}:".format(self.keys[0], self.keys[1]))
        self.modalities = modalities
        self.stored_class = study_class
        self.average_area = (self.calculate_reader_auc_average(self.modalities[self.keys[0]]),
                             self.calculate_reader_auc_average(self.modalities[self.keys[1]]))
        self.average_area_variance = (self.calculate_reader_var_average(self.modalities[self.keys[0]]),
                                      self.calculate_reader_var_average(self.modalities[self.keys[1]]))
        self.difference = self.compare_two_modalities()
        self.variance = self._two_modalities_variance()
        self.bootstrap_count = bootstrap
        self.readers = len(self.modalities[self.keys[0]])
        self.boot_variance = 0.0
        if bootstrap > 0:
            self.bootstrap_par(bootstrap)

    def __str__(self):
        """
        Displays information about the difference between the two modalities being compared.

        :return: A string with information about the modalities
        """
        string = self.name + "\n"
        string += ("DEL_A={0}\n".format(self.difference))
        if self.variance is not None:
            string += ("Del_A_Var={0}\n".format(self.variance))
        if self.bootstrap_count:
            string += ("Bootstrap_Count={0}\n".format(self.bootstrap_count))
            string += ("Bootstrap_Del_A_Var={0}\n".format(self.boot_variance))
        string += "Specific Modalities\n\n"
        for key, value in self.modalities.iteritems():
            string += ("Name: {0}\nReader Count:{1}\n".format(key, len(value)))
            string += ("Average_Area={0}\nAverage_Area_Variance={0}\n".format(self.calculate_reader_auc_average(value),
                                                                            self.calculate_reader_var_average(value)))
            for reader, result in value.iteritems():
                string += "%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                string += ("Reader: {0}\n".format(reader))
                string += (str(result))
                string += "%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                string += "\n"
        return string

    def print_to_file(self, out=None):
        """
        Print information about the difference between two modalities performance.

        :param out: The Prefix of the filename to be printed to. The file is 'out.auc'. If None, prints to 'tmp.auc'.

        :return: None
        """
        if out is None: out = 'tmp'
        out += '.auc'
        with open(out, 'w') as f:
            f.write(str(self))
        print("Output saved as: {0}".format(out))
        return None

    def display_curve(self):
        """
        Displays the curve for each reader for each modality

        :return: None
        """
        plt.figure()
        for key in self.modalities:
            for values in self.modalities[key].values():
                values.display_curve(figure=False)


class _Resampler(object):
    """
    Resample the two modalities EFROC objects to generate variance.

    Each thread of 'bootstrap_par' instantiates an instance of this object, which then determines what it acts on
    when it is called.
    Each object of this type is a function.
    Hack to avoid serializing the MRMCEFROC object but still multi-thread the code.
    """

    def __init__(self, modalities, keys, seed=None):
        self.modalities = modalities
        # only works for two, list access is slow, cache the keys in instance variables so we can access them quicker.
        # since we will call this object a lot (presumably we want a large sample size for the bootstrap, access
        self.key1 = keys[0]
        self.key2 = keys[1]
        self.seed = seed
        #If no seed is passed (which should be the normal behavior), this will seed from /dev/random if possible
        #or otherwise with the system time.
        self.gen = random.Random()

    def __call__(self,x):
        """
        Resamples each modality and reader combination to generate an average difference.
        :return: the average difference
        """
        resampled = defaultdict(dict)
        modality_average = {}
        gen = self.gen
        state = gen.getstate()
        for modality, value in self.modalities.iteritems():
            for reader, record in value.iteritems():
                # make sure the state gets set, so that the same cases are selected for each reader and modality.
                gen.setstate(state)
                #inlining the below would make it faster, but much less clear.
                resampled[modality][reader] = record.resample(gen)
                # I'd like to get the EFROC out of here so that this method works for everything.
            modality_average[modality] = MRMCStudy.calculate_reader_auc_average(resampled[modality])
        return float(modality_average[self.key1] - modality_average[self.key2])