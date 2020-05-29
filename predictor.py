#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: predictor.py
# Created Date: Thursday, May 21st 2020, 5:46:20 pm
# Author: lijiazheng
###

import logging
import json
import numpy as np
import tensorflow as tf
import time
import warnings
from collections import namedtuple
from run_ner import ner_main, NERFastPredictor
from run_ner import convert_examples_to_features, InputExample

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# do_train 微调数据
# do_predict 预测test.tsv中的文件，并输出test_result
# do_service 返回类别向量，服务模式
ModelConfig = namedtuple("ModelConfig", [
    "init_checkpoint", "vocab_file", "output_dir", "data_dir",
    "do_train", "do_eval", "do_predict", "do_service",
    "max_seq_length", "train_batch_size", "eval_batch_size", "predict_batch_size", "service_batch_size",
    "learning_rate", "num_train_epochs", "warmup_proportion",
    "save_summary_steps", "save_checkpoints_steps",
])


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


def ibatch(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    yield batch


@Singleton
class BertNER:
    """ NER model based BERT. """

    name = "bert_ner"

    def __init__(self, cfg):
        self.cfg = cfg

        self.model_config = ModelConfig(
            do_service=False,
            **self.cfg["model_config"]
        )

        if self.cfg["label_names"] == []:
            self.cfg["label_names"] = [None] * len(self.cfg["labels"])
        if len(self.cfg["labels"]) != len(self.cfg["label_names"]):
            raise Exception("labels and label_name length not match in config file.")

        self.labels = self.cfg["labels"]
        self.label_names = self.cfg["label_names"]
        self.id_label_map = {_id: label for _id, label in enumerate(self.labels)}

        logger = logging.getLogger(self.cfg["logger_file"])
        logger.addHandler(logging.FileHandler(self.cfg["logger_file"]))
        # logger.addHandler(logging.StreamHandler())
        self.logger = logger

        self.estimator, self.tokenizer = ner_main(
            self.model_config, self.cfg["bert_config"], self.labels, self.logger)

        # 进入服务模式
        self.model_config = self.model_config._replace(do_train=False,
                                                       do_eval=False,
                                                       do_predict=False,
                                                       do_service=True)

        self.fast_predictor = NERFastPredictor(self.estimator, self.model_config, self.tokenizer,
                                               self.labels, self.model_config.service_batch_size,
                                               self.model_config.max_seq_length)


    def _predict_entiteis(self, text, prediction):
        token_len = 0
        entities = []
        # 跳过[CLS]，[SEP]和word piece部分的预测标签
        tokens = list(self.tokenizer.tokenize(text))
        label_names = list(filter(lambda x: x != "_X",
                           [self.id_label_map[label_id] for label_id in prediction]))
        for i, label_name in enumerate(label_names):
            if label_name.startswith("B"):
                if token_len:
                    entities.append((i - token_len, i, tokens[i - token_len:i], label_names[i - token_len][2:]))
                token_len = 1
            elif label_name.startswith("I"):
                token_len += 1
            else:
                if token_len:
                    entities.append((i - token_len, i, tokens[i - token_len:i], label_names[i - token_len][2:]))
                    token_len = 0
        if token_len:
            entities.append((i - token_len + 1, i + 1,
                             tokens[i - token_len + 1:i + 1],
                             label_names[i - token_len + 1][2:]))

        return entities

    def predict_single(self, text):
        if not self.model_config.do_service:
            raise Exception("config.do_service is closed")
        
        start_time = time.time()
        entities = []
        for _, batch_texts in self._iNERbatch([text], self.model_config.service_batch_size):
            for _ents in self._predict_batch(batch_texts):
                entities += _ents

        self.logger.info(
            "**** NER Output*****\n"
            f"text: {text}\n"
            f"entities: {entities}\n"
            f"time elapsed: {time.time() - start_time}\n"
        )

        return entities

    def _predict_batch(self, batch_texts):
        if not self.model_config.do_service:
            raise Exception("config.do_service is closed")

        result = self.fast_predictor.predict(
            batch_texts + ["placeholder"] * (self.model_config.service_batch_size - len(batch_texts))
        )

        return [
            self._predict_entiteis(text, res) for text, res in zip(batch_texts, result)
        ]

    def _iNERbatch(self, texts, batch_size):
        doc_ids, batch_texts = [], []
        for doc_id, text in enumerate(texts):
            tokens = self.tokenizer.tokenize(text)
            for tokens in ibatch(tokens, self.model_config.max_seq_length - 2):
                doc_ids.append(doc_id)
                batch_texts.append("".join(tokens))
                if len(doc_ids) == batch_size:
                    yield doc_ids, batch_texts
                    doc_ids, batch_texts = [], []

    def predict(self, text_generator):
        texts = list(text_generator)
        entities = [[] for _ in range(len(texts))]
        for doc_ids, batch_texts in self._iNERbatch(texts, self.model_config.service_batch_size):
            for doc_id, _ents in zip(doc_ids, self._predict_batch(batch_texts)):
                entities[doc_id] += _ents

        for doc_id in range(len(texts)):
            self.logger.info(f"\n{texts[doc_id]}\n{entities[doc_id]}\n")
    
        return list(zip(texts, entities))

    def __del__(self):
        if getattr(self, "fast_prediction", None):
            self.fast_predictor.close()


if __name__ == "__main__":
    with open('./config.json') as config_file:
        cfg = json.load(config_file)

    model = BertNER(cfg)
    model.predict([
        "吴奇隆很帅啊",
        "你认识周润发吗？",
        "东风风神AX7大战启辰T90", 
        "2018年美国中期选举，你认为特朗普会下台吗？",
        "【海泰发展连续三日涨停提示风险：公司没有与创投相关的收入来源】连续三日涨停的海泰发展11月12日晚间披露风险提示公告，经公司自查，谁不想做吴彦祖？公司目前生产经营活动正常。目前，公司主营业务收入和利润来源为贸易和房产租售，没有与创投相关的收入来源，也没有科技产业投资项目。公司对应2017年每股收益的市盈率为271.95倍，截至11月12日，公司动态市盈率为2442.10倍，请投资者注意投资风险。另外，谁帅过吴彦祖？"
    ])