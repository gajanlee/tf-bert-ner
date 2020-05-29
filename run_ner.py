#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: run_ner.py
# Created Date: Tuesday, May 19th 2020, 5:54:18 pm
# Author: lijiazheng
###

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf

import bert.modeling
import bert.optimization
import bert.tokenization
# from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers

InputFeatures = collections.namedtuple("InputFeatures", ["input_ids", "input_mask", "segment_ids", "label_ids"])
InputExample = collections.namedtuple("InputExample", ["guid", "text", "label"])


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, mode="normal"):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, mode=mode)

        features.append(feature)
    return features


def get_service_examples(labels, max_seq_length, texts):
    return [
        InputExample(guid=f"{i}-service", text=text,
                     label=" ".join(labels[0] for _ in range(max_seq_length)))
        for (i, text) in enumerate(texts)
    ]


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


class NERFastPredictor:

    def _create_generator(self):
        while not self.closed:
            yield {
                "input_ids": [f.input_ids for f in self.next_features],
                "input_mask": [f.input_mask for f in self.next_features],
                "segment_ids": [f.segment_ids for f in self.next_features],
                "label_ids": [f.label_ids for f in self.next_features],
            }

    def __init__(self, estimator, config, tokenizer, label_list, batch_size, max_seq_length):
        self.estimator = estimator
        self.closed = False
        self.config = config
        self.tokenizer = tokenizer
        self.labels = label_list
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.next_features = None

        self.predictions = self.estimator.predict(input_fn=self.input_fn_builder)
        self.predict(["placeholder"] * self.batch_size)

    def input_fn_builder(self):
        return tf.data.Dataset.from_generator(
            self._create_generator,
            output_types={
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                "segment_ids": tf.int32,
                "label_ids": tf.int32,
            },
            output_shapes={
                "input_ids": (None, self.config.max_seq_length),
                "input_mask": (None, self.config.max_seq_length),
                "segment_ids": (None, self.config.max_seq_length),
                "label_ids": (None, self.config.max_seq_length),
            }
        )

    def predict(self, texts):
        service_examples = get_service_examples(self.labels, self.max_seq_length, texts)
        features = convert_examples_to_features(service_examples, self.labels, self.config.max_seq_length, self.tokenizer, mode="service")
        return self._predict(features)

    def _predict(self, features):
        self.next_features = features
        if self.batch_size != len(features):
            raise ValueError("All batches must be of the same size. batch size:" + str(self.batch_size) + " This-batch:" + str(len(features)))

        return [
            next(self.predictions) for _ in range(self.batch_size)
        ]

    def close(self):
        print("fast predictor closed.")
        self.closed = True
        next(self.predictions)


class NERProcessor:
    """Processor for the Classifcation data set."""

    def __init__(self, labels):
        self.labels = labels

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_sentence(os.path.join(data_dir, "data_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_sentence(os.path.join(data_dir, "data_dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_sentence(os.path.join(data_dir, "data_test.tsv")), "test")

    def _read_sentence(self, input_file):
        """Reads a tab separated value file."""
        lines = []
        words, labels = [], []
        for line in open(input_file, encoding="utf-8"):
            line = line.strip()
            if len(line) == 0 and len(words) > 0:
                lines.append((" ".join(words), " ".join(labels)))
                words, labels = [], []
            else:
                if len(line.split(" ")) != 2:
                    continue
                word, label = line.split(" ")
                words.append(word)
                labels.append(label)
        return lines

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode="normal"):

    label_map = {label: i for (i, label) in enumerate(label_list)}
    tokens_a = example.text.split(" ")
    label_seq = example.label.split(" ")

    tokens, labels = [], []
    if mode == "service":
        # 服务阶段，纯文本
        tokens = [token for token in tokenizer.tokenize(example.text)]
        labels = ["_X"] * len(tokens)
    elif mode == "normal":
        for word, label in zip(tokens_a, label_seq):
            pieces = tokenizer.tokenize(word)
            tokens.extend(pieces)
            labels.append(label)
            labels.extend(["_X"] * (len(pieces) - 1))
    else:
        raise ValueError(f"unexpected mode type: {mode}")

    if len(tokens) > max_seq_length - 2:  # [CLS] + [SEP]
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    label_ids = [label_map["_X"]] + [label_map[l] for l in labels] + [label_map["_X"]]

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("length: %s" % len(tokens))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature


def file_based_convert_examples_to_features(examples, label_list,
                                            max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["lengths"] = create_int_feature([feature.length])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "lengths": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):

    def model_fn(features, labels, mode, params):
        print("*** Features ***")
        for name in sorted(features.keys()):
            print("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # sequence_lengths = features["lengths"]

        print('shape of input_ids', input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")

        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps, num_warmup_steps, False)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                return {"eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids)}

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels,
                 dropout_rate=0.9, lstm_size=1, cell='lstm', num_layers=1):

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
    )

    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    # 算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    embedding_dims = embedding.shape[-1].value

    if is_training:
        embedding = tf.nn.dropout(embedding, dropout_rate)

    with tf.variable_scope("project"):
        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[embedding_dims, num_labels],
                                dtype=tf.float32, initializer=initializers.xavier_initializer())

            b = tf.get_variable("b", shape=[num_labels], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            output = tf.reshape(embedding,
                                shape=[-1, embedding_dims])  # [batch_size, embedding_dims]
            pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
    logits = tf.reshape(pred, [-1, max_seq_length, num_labels])

    # only CRF layer
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transitions",
            shape=[num_labels, num_labels],
            initializer=initializers.xavier_initializer())

        log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=labels,
            transition_params=trans,
            sequence_lengths=lengths)
        loss = tf.reduce_mean(-log_likelihood)

    pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=lengths)

    return (loss, logits, trans, pred_ids)


def ner_main(config, bert_config, labels, logger):
    """
    texts: a list of input, for do_service
    """

    processor = NERProcessor(labels)

    bert_config = modeling.BertConfig.from_dict(bert_config)

    tokenizer = tokenization.FullTokenizer(
        config.vocab_file, do_lower_case=True
    )

    if config.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (config.max_seq_length, bert_config.max_position_embeddings))

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    run_config = tf.estimator.RunConfig(
        model_dir=config.output_dir,
        save_summary_steps=config.save_summary_steps,
        save_checkpoints_steps=config.save_checkpoints_steps,
        keep_checkpoint_max=5,
        log_step_count_steps=10,
        session_config=tf.ConfigProto(log_device_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)),
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if config.do_train:
        train_examples = processor.get_train_examples(config.data_dir)
        num_train_steps = int(
            len(train_examples) / config.train_batch_size * config.num_train_epochs)
        num_warmup_steps = int(num_train_steps * config.warmup_proportion)

    label_list = processor.get_labels()

    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels=len(label_list),
                                init_checkpoint=config.init_checkpoint,
                                learning_rate=config.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": config.train_batch_size,
        },
    )

    if config.do_train:
        train_file = os.path.join(config.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, config.max_seq_length, tokenizer, train_file)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", config.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=config.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if config.do_eval:
        eval_examples = processor.get_dev_examples(config.data_dir)
        eval_file = os.path.join(config.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, config.max_seq_length, tokenizer, eval_file)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", config.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=config.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn)

        output_eval_file = os.path.join(config.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if config.do_predict:
        predict_examples = processor.get_test_examples(config.data_dir)
        predict_file = os.path.join(config.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                config.max_seq_length, tokenizer,
                                                predict_file)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", config.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=config.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(config.output_dir, "test_results.txt")

        id_to_label = {id: label for id, label in enumerate(processor.get_labels())}
        id_to_label[0] = "OUT"

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            logger.info("***** Predict results *****")
            for example, prediction in zip(predict_examples, result):
                output_line = "\t".join(example.text.split(" ")) + "\n"
                output_line += "\t".join(id_to_label[label_id] for label_id in prediction) + "\n"
                writer.write(output_line)

    return estimator, tokenizer
