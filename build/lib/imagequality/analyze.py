import argparse
from . import lroc, efroc, file_parser, study
import matplotlib.pyplot as plt
import sys


def create_and_output(input_parser, filename, output=None, **kwargs):
    try:
        report = input_parser(filename=filename[0], **kwargs)
        if output is not None:
            report.print_to_file(output)
        return report
    except IOError as e:
        study.perr("Error - Cannot find {0}".format(filename[0]))
        sys.exit(1)
    except ValueError as e:
        study.perr("Error - " + e.message)
        sys.exit(1)
    except TypeError as e:
            study.perr("Error - Attempting EFROC with missing Omega")
            sys.exit(1)

def create_and_output_mrmc(study_type,input_parser, filename,keys=None, output=None, **kwargs):
    try:
        modalities = input_parser(filenames=filename, **kwargs)
        if keys is None:
            keys = list(modalities)
        report = study_type(keys,modalities,**kwargs)
        if output is not None:
            report.print_to_file(output)
        return report
    except IOError as e:
        study.perr("Error - {0}".format(e)) #TODO: work on formatting this
        sys.exit(1)
    except ValueError as e:
        study.perr("Error - " + e.message)
        sys.exit(1)
    except TypeError as e:
            study.perr("Error - Attempting EFROC with missing Omega")
            sys.exit(1)


parser = argparse.ArgumentParser(description='Analyze LROC and EFROC studies.',
                                 usage="%(prog)s [-h] --study {lroc,efroc} "
                                       "--input/-i [FILENAME [FILENAME]] [EFROC Requirements] [Options]",
                                 add_help=False)

group = parser.add_argument_group("Required")
efroc_req = parser.add_argument_group("EFROC Required")
efroc_opt = parser.add_argument_group("EFROC Options")
lroc_opt = parser.add_argument_group("LROC Options")
optional = parser.add_argument_group("General Options")

group.add_argument('--study','-study','-s', dest='study_type',
                   help='determine the type of study to perform', required=True, choices=['roc', 'lroc', 'efroc'])
group.add_argument('--input', '-i', dest='filename', help="input filename(s) (1 or 2)", required=True, nargs='*')

efroc_req.add_argument('--OMEGA', '-O', type=int,dest='omega', metavar='Number',
                       help="Area of the object. Required for EFROC studies")

efroc_opt.add_argument('--use-spi', '-u', dest="use_signal_present_images", action="store_true",
                       help='Use signal present images in computing AUC.')
efroc_opt.add_argument('--bootstrap', '-b', type=int, metavar='Number',
                       help="Number of bootstraps to take to estimate variance of difference. "
                            "[Default = 0 (Bootstrapping not performed)]", default=0)

lroc_opt.add_argument('--multiple-modalities', '-m', action='store_true',
                      help="Flag to compare multiple modalities based on id over a number of files [Default = false]")
lroc_opt.add_argument('--label-delim', '-l')
lroc_opt.add_argument('--random-readers',action='store_true')
lroc_opt.add_argument('--exclude-readers',nargs='*')


optional.add_argument('--prefix', '-p', dest='output', help='prefix for output files. [Default = No output file]',
                      default=None)
optional.add_argument('--figure','-f', help = "display figure associated with analysis. Options are [s]how,[p]df,pn[g]."
                                              "If s is passed, the figure will be show, if p or g are passed the figure "
                                              "will be saved to the file prefix.pdf/prefix.png. More than one option can"
                                              "be passed."
                                              "",choices=['s','p','g'],nargs='*',default=None)
optional.add_argument('--quiet', '-q', help='Silence standard output.', action='store_true')
optional.add_argument("--help", "-h", action="help", help="show this help message and exit")

def main(*args):
    args = parser.parse_args(*args)
    study_type = {'roc': [lroc.ROC, lroc.MRMCROC],
                  'lroc': [lroc.LROC, lroc.MRMCLROC],
                  'efroc': [efroc.EFROC, efroc.MRMCEFROC]}[args.study_type][len(args.filename) - 1]
    dict_args = vars(args)
    del dict_args['study_type']
    #setting up how to parse the file
    input_parser = file_parser.create_parser(study_type,**dict_args)
    if len(args.filename) is 1 and not args.multiple_modalities:
        report = create_and_output(input_parser=input_parser,**dict_args)
    else:
        report = create_and_output_mrmc(study_type,input_parser=input_parser,**dict_args)
    if args.output is None:
        prefix = 'tmp'
    else:
        prefix = args.output
    if not args.quiet and report is not None:
        print(report)
    if args.figure is not None:
        report.display_curve()
        if 's' in args.figure:
            plt.show()
        if 'p' in args.figure:
            plt.savefig(prefix+'.pdf', bbox_inches='tight')
        if 'g' in args.figure:
            plt.savefig(prefix+'.png', bbox_inches='tight')

if __name__ == '__main__':
    main(sys.argv)
