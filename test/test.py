__author__ = 'cxv'

import unittest
from imagequality import lroc, efroc, file_parser
import timeit
import random
import math


class MyTestCase(unittest.TestCase):
    default_lroc_parser = file_parser.ParseAndCreateLROC()
    default_roc_parser = file_parser.ParseAndCreateLROC(cls=lroc.ROC)
    default_efroc_parser = file_parser.ParseAndCreateEFROC()
    def test_lroc_init(self):
        # test a very, very basic case.
        test = lroc.LROC([1, .5, .3, .1, .4, .5], [1, .5, .4, .3], [1, 1, 1, 1, 1, 1])
        assert isinstance(test.estimation, float)

    def test_efroc_data(self):
        l = self.default_efroc_parser('../testdat/s2-1-i0-f100-b100.scd', 50)
        self.assertAlmostEqual(.69223420807671621,l.estimation)

    def test_lroc_data(self):
        test = self.default_lroc_parser('../testdat/s2-1-i0-f200-b100.rld')
        self.assertAlmostEqual(test.estimation, .635275)
        self.assertAlmostEqual(test.localization_fraction, .71)
        self.assertAlmostEqual(math.sqrt(test.variance), .03155034)
        test2 = self.default_roc_parser('../testdat/s2-1-i0-f200-b100.rld')
        self.assertAlmostEqual(test2.estimation, .8148)
        self.assertAlmostEqual(math.sqrt(test2.variance), .0241617590833)

    def test_timing(self):
        rep = 1
        s = """\
from imagequality import file_parser
default_lroc_parser = file_parser.ParseAndCreateLROC()
test = default_lroc_parser('../testdat/s2-1-i0-f100-b100.rld')

        """
        q = """\
from imagequality import file_parser
default_efroc_parser = file_parser.ParseAndCreateEFROC()
test = default_efroc_parser('../testdat/s2-1-i0-f200-b100.scd',50)

        """
        h = """\
from imagequality import efroc
from imagequality import file_parser
s = file_parser.create_parser(efroc.MRMCEFROC)
f = s(['../testdat/s2-1-i0-f200-b100.scd','../testdat/s2-1-i8-f200-b100.scd'],50)
test = efroc.MRMCEFROC(['../testdat/s2-1-i0-f200-b100.scd','../testdat/s2-1-i8-f200-b100.scd'],f)
        """
        print("LROC Index estimator takes {0} seconds to run".format(
            timeit.timeit('test._auc_estimator()', s, number=rep) / rep))
        print("LROC variance calculator takes {0} seconds to run".format(
            timeit.timeit('test._variance_calculator()', s, number=rep) / rep))
        print("EFROC Index estimator takes {0} seconds to run".format(
            timeit.timeit('test._auc_estimator()', q, number=rep) / rep))
        print("EFROC variance calculator takes {0} seconds to run".format(
            timeit.timeit('test._variance_calculator()', q, number=rep) / rep))
        print("Bootstrapped takes {0} seconds to run for {1} bootstraps".format(
            timeit.timeit('test.bootstrap(10)', h, number=rep) / rep, 10))
        print("Bootstrap threaded takes {0} seconds to run for {1} bootstraps".format(
            timeit.timeit('test.bootstrap_par(10)', h, number=rep) / rep, 10))

    def test_lroc_and_efroc_curve(self):
        test = self.default_lroc_parser('../testdat/s2-1-i0-f100-b100.rld')
        s = self.default_efroc_parser('../testdat/s2-1-i0-f200-b100.scd', 50)
        also = self.default_roc_parser('../testdat/s2-1-i0-f200-b100.rld')
        s.display_curve()
        test.display_curve()
        also.display_curve(False)
        # plt.show() #hidden so tests run without user input
        assert True

    def test_lroc_mrmc(self):
        p = file_parser.ParseAndCreateLROC().parse_and_create_mm_file(['../testdat/s2-1-i0-f100-b100.rld', '../testdat/s2-1-i0-f200-b100.rld'])
        assert lroc.MRMCLROC.calculate_reader_auc_average(p['../testdat/s2-1-i0-f100-b100.rld']) == .59005
        assert lroc.MRMCLROC.calculate_reader_auc_average(p['../testdat/s2-1-i0-f200-b100.rld']) == .635275

    def test_lroc_compare_modalities(self):
        parser = self.default_lroc_parser.parse_and_create_mm_file
        f = parser(['../testdat/s2-1-i0-f100-b100.rld', '../testdat/s2-1-i0-f100-b100_edit.csv'])
        f = lroc.MRMCLROC(['../testdat/s2-1-i0-f100-b100.rld', '../testdat/s2-1-i0-f100-b100_edit.csv'],f)
        self.assertAlmostEqual(f.difference, -.1534166666)
        f = parser(['../testdat/s2-1-i0-f100-b100.rld', '../testdat/s2-1-i12-f100-b100.rld'])
        f = lroc.MRMCLROC(['../testdat/s2-1-i0-f100-b100.rld', '../testdat/s2-1-i12-f100-b100.rld'],f)
        self.assertAlmostEqual(f.difference, .13385)
        f = parser(['../testdat/s2-1-i0-f200-b100.rld', '../testdat/s2-1-i8-f200-b100.rld'])
        f = lroc.MRMCLROC(['../testdat/s2-1-i0-f200-b100.rld', '../testdat/s2-1-i8-f200-b100.rld'],f)
        output = """\
Difference between ../testdat/s2-1-i0-f200-b100.rld and ../testdat/s2-1-i8-f200-b100.rld:
DEL_A=0.135625
Del_A_Var=0.000555926972656
DEL_Q=0.115
Del_Q_Var=0.013158875
Specific Modalities

Name: ../testdat/s2-1-i8-f200-b100.rld
Reader Count:1
Average_Area=0.7709
Average_Area_Variance=0.0007514996905
%%%%%%%%%%%%%%%%%%%%%%%%%
Reader: 1
AL=0.7709
AL_sig=0.0274134946787

QL=0.825
QL_sig=0.0269351538433

mu_11=0.17641319
mu_12=0.140865210202
mu_21=0.00456243623116
%%%%%%%%%%%%%%%%%%%%%%%%%%

Name: ../testdat/s2-1-i0-f200-b100.rld
Reader Count:1
Average_Area=0.635275
Average_Area_Variance=0.000995424456906
%%%%%%%%%%%%%%%%%%%%%%%%%
Reader: 1
AL=0.635275
AL_sig=0.0315503479681

QL=0.71
QL_sig=0.0321663390338

mu_11=0.231488174375
mu_12=0.184639563264
mu_21=0.00702353869661
%%%%%%%%%%%%%%%%%%%%%%%%%%

"""
        assert f.__str__()==output
        self.assertAlmostEqual(f.difference, .135625)

    def test_lroc_mm(self):
        parse_and_create_mm_label = file_parser.ParseAndCreateLROC().parse_and_create_mm_label
        m = parse_and_create_mm_label(['../testdat/s2-1-testcase-f100-b100.rld'])
        assert lroc.MRMCLROC.calculate_reader_auc_average(m[0]) == .73855
        assert lroc.MRMCLROC.calculate_reader_auc_average(m[1]) == .59005

    def test_efroc_mrmc(self):
        s = self.default_efroc_parser.parse_and_create_mm_file(['../testdat/s2-1-i0-f200-b100.scd', '../testdat/s2-1-i8-f200-b100.scd'], 50,
                                              use_signal_present_images=True)
        self.assertAlmostEquals(efroc.MRMCEFROC.calculate_reader_auc_average(s['../testdat/s2-1-i0-f200-b100.scd']),
                                .64258440012)
        s = efroc.MRMCEFROC(['../testdat/s2-1-i0-f200-b100.scd','../testdat/s2-1-i8-f200-b100.scd'], s ,
                            use_signal_present_images=True)
        self.assertAlmostEqual(s.difference, .131009022)
        output = """\
Difference between ../testdat/s2-1-i0-f200-b100.scd and ../testdat/s2-1-i8-f200-b100.scd:
DEL_A=0.131009022119
Specific Modalities

Name: ../testdat/s2-1-i8-f200-b100.scd
Reader Count:1
Average_Area=0.773593422239
Average_Area_Variance=0.773593422239
%%%%%%%%%%%%%%%%%%%%%%%%%
Reader: 1
AFE=0.773593422239
AFE_sig=0.026054930547

S11=0.773653810577
S12=0.718970249062
S21=0.598617313348
%%%%%%%%%%%%%%%%%%%%%%%%%%

Name: ../testdat/s2-1-i0-f200-b100.scd
Reader Count:1
Average_Area=0.64258440012
Average_Area_Variance=0.64258440012
%%%%%%%%%%%%%%%%%%%%%%%%%
Reader: 1
AFE=0.64258440012
AFE_sig=0.0295097473802

S11=0.642670163227
S12=0.568297317949
S21=0.413119876687
%%%%%%%%%%%%%%%%%%%%%%%%%%

"""
        assert output == s.__str__()
    def test_covariance_with_self(self):
        m = self.default_lroc_parser('../testdat/s2-1-i0-f200-b100.rld')
        self.assertAlmostEquals(
            lroc.LROC.covariance(m.present, m.absent, m.success, m.present, m.absent, m.success, m.estimation,
                                 m.estimation, m._heavi), m.variance)

    def test_resample(self):
        s = self.default_efroc_parser('../testdat/s2-1-i0-f200-b100.scd', 50, use_signal_present_images=True)
        random.seed(1)
        test = s.resample()
        self.assertAlmostEquals(test.estimation, .625527933)

    def test_bootstrap_efroc(self):
        s = self.default_efroc_parser.parse_and_create_mm_file(['../testdat/s2-1-i0-f200-b100.scd','../testdat/s2-1-i8-f200-b100.scd'], 50,
                                                 use_signal_present_images=True)
        s = efroc.MRMCEFROC(['../testdat/s2-1-i0-f200-b100.scd','../testdat/s2-1-i8-f200-b100.scd'],s)
        random.seed(1)
        self.assertAlmostEquals(s._resample_and_compare(random), 0.150813575039)
        s.bootstrap(700, 12)
        self.assertAlmostEquals(.00037034490831635149, s.variance)

    def test_bootstrap_vs_parallel_bootstrap(self):
        # Test to compare the parallelized and standard bootstrapping, and show they're from the same distribution.
        s = self.default_efroc_parser.parse_and_create_mm_file(['../testdat/s2-1-i0-f200-b100.scd', '../testdat/s2-1-i8-f200-b100.scd'], 50,
                                                 use_signal_present_images=True)
        s = efroc.MRMCEFROC(['../testdat/s2-1-i0-f200-b100.scd', '../testdat/s2-1-i8-f200-b100.scd'],s,bootstrap=70)
        l = s.bootstrap_par(70)
        q = s.bootstrap(70)


    def test_cov_beta(self):
        test = self.default_lroc_parser('../testdat/s2-1-i0-f200-b100.rld')
        kargs = {'present_one':test.present,'absent_one':test.absent,'success_one':test.success,'k':test._heavi,
                 'present_two':test.present,'absent_two':test.absent,'success_two':test.success}
        self.assertAlmostEqual(.5882138888,test._cov_beta_general(dim=1,**kargs))
        self.assertAlmostEqual(.410597864322,test._cov_beta_general(dim=0,**kargs))

    def test_reader_random_var(self):
        f = self.default_lroc_parser.parse_and_create_mm_file(['../testdat/s2-1-i0-f100-b100_edit.csv','../testdat/s2-1-i0-f100-b100_edit2.csv'],random_readers=True)
        f = lroc.MRMCLROC(['../testdat/s2-1-i0-f100-b100_edit.csv','../testdat/s2-1-i0-f100-b100_edit2.csv'],f,random_readers=True)
        c = self.default_lroc_parser.parse_and_create_mm_file(['../testdat/s2-1-i0-f100-b100_edit.csv','../testdat/s2-1-i0-f100-b100_edit2.csv'],random_readers=False)
        c = lroc.MRMCLROC(['../testdat/s2-1-i0-f100-b100_edit.csv','../testdat/s2-1-i0-f100-b100_edit2.csv'],c,random_readers=False)
        v = f.variance
        self.assertAlmostEqual(c.variance,.0071064421)
        self.assertAlmostEqual(v,.00163894912097)

    def test_mm_column(self):
        parse_and_create_mm_column = file_parser.ParseAndCreateLROC(
            fieldnames=['score','localization_flag','signal_present_flag','signal_id',
                        'case_id','reader_id','method_id','label']).parse_and_create_mm_column
        f = parse_and_create_mm_column(filenames= ['../testdat/s2-mm-tid-0.csv','../testdat/s2-mm-tid-1.csv'])
        report = lroc.MRMCLROC(keys=['0','1'],modalities=f)
        self.assertAlmostEqual(report.difference,.14273333)
        self.assertAlmostEqual(report.variance,.035808922896)
        y = parse_and_create_mm_column(filenames= ['../testdat/s2-mm-tid-0.csv','../testdat/s2-mm-tid-1.csv'],exclude_readers=['1','3'])
        report = lroc.MRMCLROC(keys=['0','1'],modalities=y)
        assert report.readers is 4
        self.assertAlmostEqual(report.difference,.144575)
        self.assertAlmostEqual(report.variance,.016512375289)

    def test_different_layout(self):
        parse_and_create_lroc = file_parser.ParseAndCreateLROC(field_dict={'score':'z','localization_flag':'q',
                               'signal_present_flag':'spf','signal_id':'sid',
                               'case_id':'cid','reader_id':'rid','label':'label'})
        parse_and_create_lroc_different_order = file_parser.ParseAndCreateLROC(fieldnames=['signal_present_flag','score','localization_flag','case_id','reader_id','signal_id','label'])
        #testing csv.dictreader's ability to infer from headers
        f = parse_and_create_lroc(filename="../testdat/test-header.rld")
        self.assertAlmostEqual(f.estimation,.59005)
        f = parse_and_create_lroc(filename="../testdat/test-header-moved.csv")
        self.assertAlmostEqual(f.estimation,.59005)
        f = parse_and_create_lroc_different_order(filename="../testdat/test-no-header-moved.csv")
        self.assertAlmostEqual(f.estimation,.59005)

    def test_different_efroc_layout(self):
        list_head_dict = {'begin-list':0,'num_sig':1,'area_scanned':2,'case_id':3,'reader_id':4,'label':6,'modality_id':5}
        parse_and_create_efroc = file_parser.create_parser(efroc.MRMCEFROC,multiple_modalities=True,list_head_dict=list_head_dict)
        s = parse_and_create_efroc(['../testdat/s2-1-test-case.scd'],omega=50)
        r = efroc.MRMCEFROC(list(s),s)
        assert r.difference == 0.0

if __name__ == '__main__':
    unittest.main()
