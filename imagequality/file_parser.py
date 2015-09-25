__author__ = 'cxv'
import csv
from collections import defaultdict
from . import lroc,efroc

def create_parser(study_type,label_delim=None,multiple_modalities=False,**kwargs):
    if study_type is lroc.LROC or study_type is lroc.ROC:
        if multiple_modalities:
            return ParseAndCreateLROC(cls=study_type,**kwargs).parse_and_create_mm_column
        return ParseAndCreateLROC(study_type,**kwargs)
    elif study_type is efroc.EFROC:
        if multiple_modalities:
            return ParseAndCreateEFROC(cls=study_type,**kwargs).parse_and_create_mm_column
        return ParseAndCreateEFROC(study_type,**kwargs)
    elif study_type is lroc.MRMCLROC or study_type is lroc.MRMCROC:
        cls = lroc.LROC if study_type is lroc.MRMCLROC else lroc.ROC
        if label_delim is not None:
            return ParseAndCreateLROC(cls=cls,label_delim=label_delim,**kwargs).parse_and_create_mm_label
        if multiple_modalities:
            return ParseAndCreateLROC(cls=cls,**kwargs).parse_and_create_mm_column
        return ParseAndCreateLROC(cls=cls,**kwargs).parse_and_create_mm_file
    elif study_type is efroc.MRMCEFROC:
        if multiple_modalities:
            return ParseAndCreateEFROC(cls=efroc.EFROC,**kwargs).parse_and_create_mm_column
        return ParseAndCreateEFROC(cls=efroc.EFROC,**kwargs).parse_and_create_mm_file

class ParseAndCreateLROC(object):
    """
    Callable which parses an arbitrary number of files and creates a dictionary keyed to
    filename which contains a dictionary keyed to reader id which contains LROC objects.

    :param filenames: list of filenames corresponding to files that can be parsed to LROC
    :return: a dictionary of dictionaries of lrocs
    """
    # defines a dictionary of present, absent and success. Each will be a dictionary with key equal to the reader id,
    # and value being an m or n length list, like something that will be passed to the LROC constructor.
    # these get constructed as lroc objects, which are stored in the modalities dictionary.
    # this is super generic

    def __init__(self,cls=None,fieldnames = None, present_ident = "1", absent_ident = "0",field_dict=None,label_delim=None,comment_string="#",**kwargs):
        self.present_ident = present_ident
        self.absent_ident = absent_ident
        self.comment_string = comment_string
        if field_dict is not None:
            self.field_dict = field_dict
        else:
            self.field_dict = {'score':'score','localization_flag':'localization_flag',
                               'signal_present_flag':'signal_present_flag','signal_id':'signal_id',
                               'case_id':'case_id','reader_id':'reader_id','label':'label','method_id':'method_id'}
        if cls is None:
            self.cls = lroc.LROC
        else:
            self.cls = cls
        if fieldnames is not None:
            self.fieldnames = fieldnames
        else:
            self.fieldnames = ['score','localization_flag','signal_present_flag','signal_id','case_id','reader_id','label']
        if label_delim is None:
            self.label_delim = 'fbp'
        else:
            self.label_delim = label_delim

    def parse_and_create_mm_file(self, filenames, exclude_readers=None, **kwargs):
        modalities = {}
        field_dict = self.field_dict
        present_identifier = self.present_ident
        absent_identifier = self.absent_ident
        if exclude_readers is None:
            exclude_readers = []
        for filename in filenames:
            lroc_readers = {}
            present = defaultdict(list)
            absent = defaultdict(list)
            success = defaultdict(list)

            with open(filename) as f:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
                header = csv.Sniffer().has_header(f.read(1024))
                f.seek(0)
                fieldnames = self.fieldnames if not header else None
                reader = csv.DictReader(f, dialect=dialect,fieldnames=fieldnames,restkey="runover",restval="")
                try:
                    for row in reader:
                        if row[reader.fieldnames[0]] == self.comment_string:
                            continue
                        if row[field_dict['signal_present_flag']] == present_identifier:
                            present[row[field_dict['reader_id']]].append(float(row[field_dict['score']]))
                            success[row[field_dict['reader_id']]].append(float(row[field_dict['localization_flag']]))
                        elif row[field_dict['signal_present_flag']] == absent_identifier:
                            absent[row[field_dict['reader_id']]].append(float(row[field_dict['score']]))
                        else:
                            raise ValueError("Present signals should be identified with {0}, absent signals with {1}".format(present_identifier,absent_identifier))
                except ValueError:
                    raise ValueError(("Missing input at line {0} in file {1}".format(reader.line_num, filename)))
                for reader_id in present:
                    if reader_id not in exclude_readers:
                        lroc_readers[reader_id] = self.cls(present[reader_id], absent[reader_id], success[reader_id],
                                                               name=(filename[:-4] + "-reader_{0}".format(reader_id)))
                        modalities[filename] = lroc_readers
        return modalities

    def __call__(self, filename, **kwargs):
        """
        Parses a file, creates an LROC object from it, and prints that object to a specific file.
        :param filename: The filename to be parsed. The way that the file should be formatted is specified in the
        constructor of this method.
        :param print_to_file: True if the lroc should be printed to a file
        :param out: the prefix of the filename to print the lroc to. defaults to tmp
        :return: The created LROC instance
        """
        present = []
        absent = []
        success = []
        field_dict = self.field_dict
        present_identifier = self.present_ident
        absent_identifier = self.absent_ident
        with open(filename) as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            header = csv.Sniffer().has_header(f.read(1024))
            f.seek(0)
            fieldnames = self.fieldnames if not header else None
            reader = csv.DictReader(f, dialect=dialect,fieldnames=fieldnames,restkey="runover",restval="")
            try:
                for row in reader:
                    if row[reader.fieldnames[0]] == self.comment_string:
                        continue
                    if row[field_dict['signal_present_flag']] == present_identifier:
                        present.append(float(row[field_dict['score']]))
                        success.append(float(row[field_dict['localization_flag']]))
                    elif row[field_dict['signal_present_flag']] == absent_identifier:
                        absent.append(float(row[field_dict['score']]))
                    else:
                        raise ValueError("Present signals should be identified with {0}, absent signals with {1}".format(present_identifier,absent_identifier))
            except ValueError:
                raise ValueError(("Missing input at line {0} in file {1}".format(reader.line_num, filename)))
        return_lroc = self.cls(present, absent, success, name=filename[:-4])
        return return_lroc

    def parse_and_create_mm_column(self, filenames, exclude_readers=None, **kwargs):
        ##Needs to have 'method column' somewhere.
        present = defaultdict(lambda: defaultdict(list))
        absent = defaultdict(lambda: defaultdict(list))
        success = defaultdict(lambda: defaultdict(list))
        modalities = defaultdict(lambda: defaultdict())
        field_dict = self.field_dict
        present_identifier = self.present_ident
        absent_identifier = self.absent_ident
        if exclude_readers is None:
            exclude_readers = []
        for filename in filenames:
            with open(filename) as f:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
                header = csv.Sniffer().has_header(f.read(1024))
                f.seek(0)
                fieldnames = self.fieldnames if not header else None
                reader = csv.DictReader(f, dialect=dialect,fieldnames=fieldnames,restkey="label",restval="")
                try:
                    for row in reader:
                        if row[reader.fieldnames[0]] == self.comment_string:
                            continue
                        if row[field_dict['signal_present_flag']] == present_identifier:
                            present[row[field_dict['method_id']]][row[field_dict['reader_id']]].append(float(row[field_dict['score']]))
                            success[row[field_dict['method_id']]][row[field_dict['reader_id']]].append(float(row[field_dict['localization_flag']]))
                        elif row[field_dict['signal_present_flag']] == absent_identifier:
                            absent[row[field_dict['method_id']]][row[field_dict['reader_id']]].append(float(row[field_dict['score']]))
                        else:
                            raise ValueError("Present signals should be identified with {0}, "
                                             "absent signals with {1}".format(present_identifier,absent_identifier))
                except ValueError:
                    raise ValueError(("Missing input at line {0} in file {1}".format(reader.line_num, filename)))
        for method_id in present.keys():
                        for reader_id in present[method_id].keys():
                            if reader_id not in exclude_readers:
                                modalities[method_id][reader_id] = self.cls(present[method_id][reader_id],
                                                                        absent[method_id][reader_id],
                                                                        success[method_id][reader_id],
                                                                        name="TID{0}RID{1}".format(method_id,reader_id))
        return modalities

    def parse_and_create_mm_label(self, filenames,label_delim=None,exclude_readers=None, **kwargs):
        present = defaultdict(lambda: defaultdict(list))
        absent = defaultdict(lambda: defaultdict(list))
        success = defaultdict(lambda: defaultdict(list))
        modalities = defaultdict(lambda: defaultdict())
        field_dict = self.field_dict
        present_identifier = self.present_ident
        absent_identifier = self.absent_ident
        if exclude_readers is None:
            exclude_readers = []
        if label_delim is None:
            label_delim = self.label_delim
        for filename in filenames:
            with open(filename) as f:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
                header = csv.Sniffer().has_header(f.read(1024))
                f.seek(0)
                fieldnames = self.fieldnames if not header else None
                reader = csv.DictReader(f, dialect=dialect,fieldnames=fieldnames,restkey="extra",restval="")
                try:
                    for row in reader:
                        if row[self.fieldnames[0]] == self.comment_string:
                            continue
                        contains_delim = label_delim in row[field_dict['label']]
                        if row[field_dict['signal_present_flag']] == present_identifier:
                            present[contains_delim][row[field_dict['reader_id']]].append(float(row[field_dict['score']]))
                            success[contains_delim][row[field_dict['reader_id']]].append(float(row[field_dict['localization_flag']]))
                        elif row[field_dict['signal_present_flag']] == absent_identifier:
                            absent[contains_delim][row[field_dict['reader_id']]].append(float(row[field_dict['score']]))
                        else:
                            raise ValueError("Present signals should be identified with {0}, "
                                             "absent signals with {1}".format(present_identifier,absent_identifier))
                except ValueError:
                    raise ValueError(("Missing input at line {0} in file {1}".format(reader.line_num, filename)))
        for method_id in present.keys():
                        for reader_id in present[method_id].keys():
                            if reader_id not in exclude_readers:
                                string_method_id = label_delim if method_id else "not-" + label_delim
                                modalities[method_id][reader_id] = self.cls(present[method_id][reader_id],
                                                                             absent[method_id][reader_id],
                                                                             success[method_id][reader_id],
                                                                             name=string_method_id + reader_id)

        return modalities

class ParseAndCreateEFROC(object):
    def __call__(self, filename,omega,use_signal_present_images=False,reader_id=1,**kwargs):
        return self.parse_and_create_mm_file([filename],omega,use_signal_present_images,**kwargs)[filename][reader_id]

    def __init__(self,cls=None,present_ident = "1", absent_ident = "0",list_head_dict=None,list_body_dict = None,
                list_delim = "begin-list",label_delim="fbp",**kwargs):
        self.present_ident = present_ident
        self.absent_ident = absent_ident
        self.label_delim = label_delim
        self.list_delim = list_delim
        if cls is None:
            self.cls = efroc.EFROC
        else:
            self.cls = cls
        if list_head_dict is not None:
            self.list_head_dict = list_head_dict
        else:
            self.list_head_dict = {'begin-list':0,'num_sig':1,'area_scanned':2,'case_id':3,'reader_id':4,'label':5}
        if list_body_dict is not None:
            self.list_body_dict = list_body_dict
        else:
            self.list_body_dict = {'score':0,'signal_localization_flag':1,'signal_id':2}

    def parse_and_create_mm_file(self, filenames, omega, use_signal_present_images=False,**kwargs):
        """
        Parse an arbitrary number of files and create a dictionary of dictionary of EFROC objects.

        :param args: Filenames and omega for each instance to be parsed. Expected as filename1,omega1,filename2,omega2...
        :param kargs: use_spi, which determines if improperly localized marks from signal present images should be used
            for false signal scores. Defaults to true.

        :return: A dictionary with filename as keys, with dictionaries as values. In those dictionaries, reader id's are
            keys and values are EFROC objects
        """
        modalities = {}
        for filename in filenames:
            report = {}
            true = defaultdict(defaultdict)
            false = defaultdict(defaultdict)
            area = defaultdict(defaultdict)
            count_signals = defaultdict(dict)
            signal_id = defaultdict(list)
            with open(filename) as f:
                case_id = 0
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[self.list_head_dict['begin-list']].lower() == self.list_delim:
                        case_id += 1
                        reader_id = int(row[self.list_head_dict['reader_id']])
                        num_sig = int(row[self.list_head_dict['num_sig']])
                        count_signals[reader_id][case_id] = num_sig
                        area[reader_id][case_id] = (float(row[self.list_head_dict['area_scanned']]))
                        true[reader_id][case_id] = []
                        false[reader_id][case_id] = []
                        for list_body in reader:  # uses the same iterator as the loop it is contained in.
                            if list_body[0].lower() == "end-list":
                                break
                            score = list_body[self.list_body_dict['score']]
                            signal_localization_flag = list_body[self.list_body_dict['signal_localization_flag']]
                            if num_sig > 0:
                                if signal_localization_flag == self.present_ident:
                                    true[reader_id][case_id].append(float(score))
                                elif signal_localization_flag == self.absent_ident and use_signal_present_images:
                                    false[reader_id][case_id].append(float(score))
                            else:
                                false[reader_id][case_id].append(float(score))
                            try:
                                signal_id[reader_id][case_id].append(int(list_body[self.list_body_dict['signal_id']]))
                            except IndexError:
                                pass
                    else:
                        raise ValueError("no {0} at line {1} in file {2}".format(self.list_delim,reader.line_num, filename))
                for reader_id in true:
                    report[reader_id] = self.cls(area[reader_id], omega, true[reader_id], false[reader_id],
                                             count_signals[reader_id], name=filename[:-4] + "-r{0}".format(reader_id))
                modalities[filename] = report
        return modalities

    def parse_and_create_mm_column(self, filenames, omega, use_signal_present_images=False,**kwargs):
        """
        Parse an arbitrary number of files and create a dictionary of dictionary of EFROC objects.

        :param args: Filenames and omega for each instance to be parsed. Expected as filename1,omega1,filename2,omega2...
        :param kargs: use_spi, which determines if improperly localized marks from signal present images should be used
            for false signal scores. Defaults to true.

        :return: A dictionary with filename as keys, with dictionaries as values. In those dictionaries, reader id's are
            keys and values are EFROC objects
        """
        modalities = {}
        for filename in filenames:
            report = defaultdict(dict)
            true = defaultdict(lambda: defaultdict(defaultdict))
            false = defaultdict(lambda: defaultdict(defaultdict))
            area = defaultdict(lambda: defaultdict(defaultdict))
            count_signals = defaultdict(lambda: defaultdict(defaultdict))
            signal_id = defaultdict(lambda: defaultdict(defaultdict))
            with open(filename) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[self.list_head_dict['begin-list']].lower() == self.list_delim:
                        case_id = int(row[self.list_head_dict['case_id']])
                        reader_id = int(row[self.list_head_dict['reader_id']])
                        num_sig = int(row[self.list_head_dict['num_sig']])
                        modality_id = int(row[self.list_head_dict['modality_id']])
                        count_signals[modality_id][reader_id][case_id] = num_sig
                        area[modality_id][reader_id][case_id] = (float(row[self.list_head_dict['area_scanned']]))
                        true[modality_id][reader_id][case_id] = []
                        false[modality_id][reader_id][case_id] = []
                        for list_body in reader:  # uses the same iterator as the loop it is contained in.
                            if list_body[0].lower() == "end-list":
                                break
                            score = list_body[self.list_body_dict['score']]
                            signal_localization_flag = list_body[self.list_body_dict['signal_localization_flag']]
                            if num_sig > 0:
                                if signal_localization_flag == self.present_ident:
                                    true[modality_id][reader_id][case_id].append(float(score))
                                elif signal_localization_flag == self.absent_ident and use_signal_present_images:
                                    false[modality_id][reader_id][case_id].append(float(score))
                            else:
                                false[modality_id][reader_id][case_id].append(float(score))
                            try:
                                signal_id[modality_id][reader_id][case_id].append(int(list_body[self.list_body_dict['signal_id']]))
                            except (IndexError,KeyError):
                                pass
                    else:
                        raise ValueError("no {0} at line {1} in file {2}".format(self.list_delim,reader.line_num, filename))
            for modality_id in false:
                for reader_id in true[modality_id]:
                    report[modality_id][reader_id] = self.cls(area[modality_id][reader_id], omega,
                                                              true[modality_id][reader_id], false[modality_id][reader_id],
                                                              count_signals[modality_id][reader_id],name=("modality{0}-reader{1}".format(modality_id,reader_id)))
            modalities = report
        return modalities