import json
import os
import random

from ..base.base_client_data_loader import BaseClientDataLoader
from ..base.base_raw_data_loader import BaseRawDataLoader


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "span_extraction"
        self.document_X = []
        self.question_X = []
        self.attributes = dict()
        self.train_file_name = "train-v1.1.json"
        self.test_file_name = "dev-v1.1.json"

    def data_loader(self):
        if len(self.document_X) == 0 or len(self.question_X) == 0 or len(self.Y) == 0:
            context_X, question_X, Y = self.process_data(os.path.join(self.data_path, self.train_file_name))
            train_size = len(context_X)
            temp = self.process_data(os.path.join(self.data_path, self.test_file_name))
            context_X.extend(temp[0])
            question_X.extend(temp[1])
            Y.extend(temp[2])
            train_index_list = [i for i in range(train_size)]
            test_index_list = [i for i in range(train_size, len(context_X))]
            index_list = train_index_list + test_index_list
            self.context_X, self.question_X, self.Y = context_X, question_X, Y
            self.attributes["train_index_list"] = train_index_list
            self.attributes["test_index_list"] = test_index_list
            self.attributes["index_list"] = index_list
        return {"context_X": self.context_X, "question_X": self.question_X, "Y": self.Y, "attributes": self.attributes,
                "task_type": self.task_type}

    def process_data(self, file_path):
        context_X = []
        question_X = []
        Y = []
        if "doc_index" not in self.attributes:
            self.attributes["doc_index"] = []
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)

            for index, document in enumerate(data["data"]):
                for paragraph in document["paragraphs"]:
                    for qas in paragraph["qas"]:
                        answers_index = []
                        answers_text = []
                        context_X.append(paragraph["context"])
                        question_X.append(qas["question"])
                        for answer in qas["answers"]:
                            if answer["text"] not in answers_text:
                                answers_text.append(answer["text"])
                                start = answer["answer_start"]
                                end = start + len(answer["text"].rstrip())
                                answers_index.append((start, end))
                        Y.append(answers_index)
                        self.attributes["doc_index"].append(index)

        return context_X, question_X, Y

    # TODO: Unified Partition Interface
    @staticmethod
    def nature_partition(attributes):
        train_doc_index_set = set([attributes["doc_index"][i] for i in attributes["train_index_list"]])
        partition_dict = dict()
        partition_dict["partition_data"] = dict()
        partition_dict["n_clients"] = len(train_doc_index_set)
        for doc_id in train_doc_index_set:
            for i in attributes["train_index_list"]:
                if attributes["doc_index"][i] == doc_id:
                    if doc_id not in partition_dict["partition_data"]:
                        partition_dict["partition_data"][doc_id] = dict()
                        partition_dict["partition_data"][doc_id]["train"] = list()
                        partition_dict["partition_data"][doc_id]["test"] = list()
                    partition_dict["partition_data"][doc_id]["train"].append(i)

        test_doc_index_set = set([attributes["doc_index"][i] for i in attributes["test_index_list"]])
        for doc_id in test_doc_index_set:
            test_doc_index_list = []
            for i in attributes["test_index_list"]:
                if attributes["doc_index"][i] == doc_id:
                    test_doc_index_list.append(i)
            client_idx = random.randint(0, partition_dict["n_clients"] - 1)
            partition_dict["partition_data"][client_idx]["test"].extend(test_doc_index_list)

        return partition_dict


class ClientDataLoader(BaseClientDataLoader):

    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ["context_X", "question_X", "Y"]
        attribute_fields = []
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields,
                         attribute_fields)
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        tokenizer = self.spacy_tokenizer.en_tokenizer

        def __tokenize_data(data):
            for i in range(len(data["context_X"])):
                data["context_X"][i] = [token.text.strip().lower() for token in tokenizer(data["context_X"][i].strip()) if token.text.strip()]
                data["question_X"][i] = [token.text.strip().lower() for token in tokenizer(data["question_X"][i].strip()) if token.text.strip()]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)
