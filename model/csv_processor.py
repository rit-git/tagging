from transformers.data.processors.glue import Sst2Processor
from transformers.data.processors.glue import InputExample
import csv
import os
import sys
class CsvSst2Processor(Sst2Processor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._csv_create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        path = os.path.join(data_dir, "valid.csv")
        if not os.path.exists(path):
            path = os.path.join(data_dir, "dev.csv")

        return self._csv_create_examples(
            self._read_csv(path), "valid")


    def get_test_examples(self, data_dir):
        """See base class."""
        return self._csv_create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_labels_multi(self, n = 3):
        """See base class."""
        label_list = []
        for i in range(n):
            label_list.append(str(i))
            
        return label_list 

    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            lines = []
            for line in reader:
                assert(len(line) == 3)
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _csv_create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            assert(len(line) == 3)
            text_a = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

csv_processors = {
    "sst-2": CsvSst2Processor,
}

#processor = csv_processors['sst-2']
