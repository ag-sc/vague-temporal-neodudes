import csv
import json
import os
import compress_pickle as cpl
import random
import sys
from typing import List, Optional, Dict, Set

import lightning as L
import more_itertools
import torch
import torch.nn.functional as F
import transformers
import lemon
import pandas as pd

from torch.utils.data import random_split, DataLoader, Dataset

from dudes import utils, consts
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint


class QueryScoreDataset(Dataset):
    def __init__(self, data: List, pad_token_id: int, pad_multiplier: int = 512):
        self.data = data
        self.pad_token_id = pad_token_id
        self.pad_multiplier = pad_multiplier
        input_max_length = max([len(de["input1"]) for de in data] + [len(de["input2"]) for de in data])
        self.input_target_length = 512 * (input_max_length // self.pad_multiplier + (0 if input_max_length % self.pad_multiplier == 0 else 1))
        print("self.input_target_length", self.input_target_length, flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        de: Dict[str, str] = self.data[idx]
        input1: torch.Tensor = torch.LongTensor(de["input1"])
        input1 = F.pad(input1, (0, self.input_target_length - len(de["input1"])), value=self.pad_token_id)
        input2: torch.Tensor = torch.LongTensor(de["input2"])
        input2 = F.pad(input2, (0, self.input_target_length - len(de["input2"])), value=self.pad_token_id)
        output: torch.Tensor = torch.Tensor(de["target"])

        return {
            "input_token_ids": torch.cat((input1, input2)),
            "output_values": output,
        }

class FuzzyQueryScoreDataModule(L.LightningDataModule):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 train_json_path: Optional[str] = None,
                 val_json_path: Optional[str] = None,
                 test_json_path: Optional[str] = None,
                 all_queries_csv_path: Optional[str] = None,
                 endpoint_data_path: Optional[str] = None,
                 batch_size: int = 32,
                 combs_per_question: int = 100):
        super().__init__()
        self.query_score_train: Optional[Dataset] = None
        self.query_score_val: Optional[Dataset] = None
        self.query_score_test: Optional[Dataset] = None
        self.all_queries_csv_path = all_queries_csv_path if all_queries_csv_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "2025-05-07-full-final-evaluation_data_results.csv.zst")
        self.train_json_path = train_json_path if train_json_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "evaluation_data_train.json")
        self.val_json_path = val_json_path if val_json_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "evaluation_data_val.json")
        self.test_json_path = test_json_path if test_json_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "evaluation_data_test.json")
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.combs_per_question = combs_per_question
        self.query_score_train = None
        self.query_score_val = None
        self.query_score_test = None
        self.endpoint_data_path = endpoint_data_path if endpoint_data_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "household_events.ttl")
        self.cache = dict()
        self.se = SPARQLEndpoint(endpoint=self.endpoint_data_path, cache=self.cache)

        # self.input_template = "Question: {question}" + self.tokenizer.pad_token
        # self.input_template += "SPARQL: {sparql1} DUDES: {dudes1} Number of results: {numres1}" + self.tokenizer.pad_token
        # self.input_template += "SPARQL: {sparql2} DUDES: {dudes2} Number of results: {numres2}"
    def prepare_data(self):
        if not os.path.isfile(self.train_json_path+f"-dataset.cpkl"):
            df = pd.read_csv(self.all_queries_csv_path)
            df1 = df[df["F1"] >= 0.01]
            df2 = df[df["F1"] < 0.01]
            train_data = json.load(open(self.train_json_path))
            train_ids = {int(d["id"]) for d in train_data}

            float_func = self._get_f1

            train_data_raw = []
            query_set = set()
            td, qs = self._gen_data(df1, train_ids, float_func=float_func)
            train_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data(df2, train_ids, float_func=float_func)
            train_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data_comb(df1, df2, train_ids, float_func=float_func)
            train_data_raw += td
            query_set.update(qs)

            with open(self.train_json_path+f"-dataset.cpkl", "wb") as f:
                cpl.dump(train_data_raw, f, compression="gzip")

        if not os.path.isfile(self.val_json_path+f"-dataset.cpkl"):
            df = pd.read_csv(self.all_queries_csv_path)
            val_data = json.load(open(self.val_json_path))
            valid_ids = {int(d["id"]) for d in val_data}
            valid_data_raw = []
            query_set = set()
            td, qs = self._gen_data(df1, valid_ids, float_func=float_func)
            valid_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data(df2, valid_ids, float_func=float_func)
            valid_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data_comb(df1, df2, valid_ids, float_func=float_func)
            valid_data_raw += td
            query_set.update(qs)

            with open(self.val_json_path+f"-dataset.cpkl", "wb") as f:
                cpl.dump(valid_data_raw, f, compression="gzip")

        if not os.path.isfile(self.test_json_path+f"-dataset.cpkl"):
            df = pd.read_csv(self.all_queries_csv_path)
            df1 = df[df["F1"] >= 0.01]
            df2 = df[df["F1"] < 0.01]
            test_data = json.load(open(self.test_json_path))
            test_ids = {int(d["id"]) for d in test_data}
            test_data_raw = []
            query_set = set()
            td, qs = self._gen_data(df1, test_ids, float_func=float_func)
            test_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data(df2, test_ids, float_func=float_func)
            test_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data_comb(df1, df2, test_ids, float_func=float_func)
            test_data_raw += td
            query_set.update(qs)

            with open(self.test_json_path+f"-dataset.cpkl", "wb") as f:
                cpl.dump(test_data_raw, f, compression="gzip")


    def _input_string(self, de1: Dict[str, str]):
        numres = 0
        # try:
        #     self.se.get_results_query(de1["SPARQL Query"])
        #     numres = len(self.se.get_results_query(de1["SPARQL Query"]))
        # except Exception as e:
        #     print(f"Error in SPARQL endpoint: {e}")
        if de1["Prediction"] == "Yes" or de1["Prediction"] == "No":
            numres = 1
        elif len(de1["Prediction"]) != "[]":
            numres = de1["Prediction"].count(",") + 1

        return """Question: {question}
            Number of results: {numres1}
            SPARQL: 
            {sparql1} 
            DUDES: 
            {dudes1}""".format(
            question=de1["Question"],
            numres1=numres,
            sparql1=utils.replace_namespaces_dirty(utils.remove_prefix(de1["SPARQL Query"])),
            dudes1=de1["DUDES"],
        )

    @staticmethod
    def _get_f1(de) -> float:
        if "FILTER" in de['SPARQL Query'] and (("What" in de["Question"] and "UNION" in de["SPARQL Query"]) or ("What" not in de["Question"] and "UNION" not in de["SPARQL Query"])):
            return float(de["F1"])
        else:
            return 0.0

    def _gen_data(self, df: pd.DataFrame, valid_ids: Optional[set] = None, float_func=None):
        if float_func is None:
            float_func = self._get_f1
        train_data_raw = []
        query_ids = set()
        for q, indices in df.groupby('Question').groups.items():
            if len(indices) < 2 or (valid_ids is not None and df.loc[indices[0]]["ID"] not in valid_ids):
                continue

            for i in range(self.combs_per_question):
                row_id1, row_id2 = more_itertools.random_combination(indices, 2)
                query_ids.add(row_id1)
                query_ids.add(row_id2)
                de1 = df.loc[row_id1]
                de2 = df.loc[row_id2]
                ede1 = self.tokenizer.encode(self._input_string(de1), max_length=512, truncation=True)
                ede2 = self.tokenizer.encode(self._input_string(de2), max_length=512, truncation=True)

                train_data_raw.append({
                    "input1": ede1,
                    "input2": ede2,
                    "target": [float_func(de1), float_func(de2)],
                    # "target": [(float(de1["F1"]) / 2.0) - (float(de2["F1"]) / 2.0) + 0.5],
                    # norm to 0-1 with 0 = left, 1 = right
                })
                train_data_raw.append({
                    "input1": ede2,
                    "input2": ede1,
                    "target": [float_func(de2), float_func(de1)],
                    # "target": [(float(de2["F1"]) / 2.0) - (float(de1["F1"]) / 2.0) + 0.5],
                    # norm to 0-1 with 0 = left, 1 = right
                })
        return train_data_raw, query_ids

    def _gen_data_comb(self, df1: pd.DataFrame, df2: pd.DataFrame, valid_ids: Optional[set] = None, float_func=None):
        if float_func is None:
            float_func = self._get_f1
        train_data_raw = []
        query_ids = set()
        for q, indices1 in df1.groupby('Question').groups.items():
            if len(indices1) < 2 or (valid_ids is not None and df1.loc[indices1[0]]["ID"] not in valid_ids):
                continue
            gr2 = df2.groupby('Question').groups
            if q not in gr2:
                continue
            indices2 = gr2[q]
            for i in range(2 * self.combs_per_question):
                # row_id1, row_id2 = more_itertools.random_combination(indices, 2)
                row_id1 = random.choice(indices1)
                row_id2 = random.choice(indices2)
                query_ids.add(row_id1)
                query_ids.add(row_id2)
                de1 = df1.loc[row_id1]
                de2 = df2.loc[row_id2]
                ede1 = self.tokenizer.encode(self._input_string(de1), max_length=512, truncation=True)
                ede2 = self.tokenizer.encode(self._input_string(de2), max_length=512, truncation=True)

                train_data_raw.append({
                    "input1": ede1,
                    "input2": ede2,
                    "target": [float_func(de1), float_func(de2)],
                    # "target": [(float(de1["F1"]) / 2.0) - (float(de2["F1"]) / 2.0) + 0.5],
                    # norm to 0-1 with 0 = left, 1 = right
                })
                train_data_raw.append({
                    "input1": ede2,
                    "input2": ede1,
                    "target": [float_func(de2), float_func(de1)],
                    # "target": [(float(de2["F1"]) / 2.0) - (float(de1["F1"]) / 2.0) + 0.5],
                    # norm to 0-1 with 0 = left, 1 = right
                })
        return train_data_raw, query_ids

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # if stage == "fit" or stage is None:
        if self.query_score_train is None:
            with open(self.train_json_path+f"-dataset.cpkl", "rb") as f:
                train_data_raw = cpl.load(f, compression="gzip")

            train_dataset = QueryScoreDataset(train_data_raw, self.tokenizer.pad_token_id)

            self.query_score_train = train_dataset
        if self.query_score_val is None:
            with open(self.val_json_path+f"-dataset.cpkl", "rb") as f:
                valid_data_raw = cpl.load(f, compression="gzip")

            valid_dataset = QueryScoreDataset(valid_data_raw, self.tokenizer.pad_token_id)

            self.query_score_val = valid_dataset

            # if self.seed is None:
            #     self.query_score_train, self.query_score_val = random_split(
            #         full_dataset, [0.9, 0.1]
            #     )
            # else:
            #     self.query_score_train, self.query_score_val = random_split(
            #         full_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(self.seed)
            #     )

            # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage == "predict":
        if self.query_score_test is None:
            with open(self.test_json_path+f"-dataset.cpkl", "rb") as f:
                test_data_raw = cpl.load(f, compression="gzip")

            test_dataset = QueryScoreDataset(test_data_raw, self.tokenizer.pad_token_id)

            self.query_score_test = test_dataset

    def train_dataloader(self):
        assert self.query_score_train is not None
        return DataLoader(self.query_score_train, shuffle=True, batch_size=self.batch_size)  # , num_workers=8)

    def val_dataloader(self):
        assert self.query_score_val is not None
        return DataLoader(self.query_score_val, shuffle=False, batch_size=self.batch_size)  # , num_workers=8)

    def test_dataloader(self):
        assert self.query_score_test is not None
        return DataLoader(self.query_score_test, shuffle=False, batch_size=self.batch_size)  # , num_workers=8)

    def predict_dataloader(self):
        assert self.query_score_test is not None
        return DataLoader(self.query_score_test, shuffle=False, batch_size=self.batch_size)  # , num_workers=8)