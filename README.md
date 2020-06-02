# BERT-NER

### Usage

* 把run_ner.py放在bert目录下
* 使用predictor读取配置文件，并进行训练、预测
* 测试命令【service模式输出】：`python predictor.py`
```
2020-05-29 17:45:00,886 - ./output/ner.log - INFO - 
吴奇隆很帅啊
[(0, 3, ['吴', '奇', '隆'], 'PER')]

2020-05-29 17:45:00,886 - ./output/ner.log - INFO - 
你认识周润发吗？
[(3, 6, ['周', '润', '发'], 'PER')]

2020-05-29 17:45:00,886 - ./output/ner.log - INFO - 
东风风神AX7大战启辰T90
[]

2020-05-29 17:45:00,886 - ./output/ner.log - INFO - 
2018年美国中期选举，你认为特朗普会下台吗？
[(12, 15, ['特', '朗', '普'], 'PER')]

2020-05-29 17:45:00,886 - ./output/ner.log - INFO - 
【海泰发展连续三日涨停提示风险：公司没有与创投相关的收入来源】连续三日涨停的海泰发展11月12日晚间披露风险提示公告，经公司自查，谁不想做吴彦祖？公司目前生产经营活动正常。目前，公司主营业务收入和利润来源为贸易和房产租售，没有与创投相关的收入来源，也没有科技产业投资项目。公司对应2017年每股收益的市盈率为271.95倍，截至11月12日，公司动态市盈率为2442.10倍，请投资者注意投资风险。另外，谁帅过吴彦祖？
[(5, 8, ['吴', '彦', '祖'], 'PER'), (6, 9, ['吴', '彦', '祖'], 'PER')]
```
* 运行环境
    - tensorflow[-gpu] == 1.12.0

### 数据

* run.sh中提供三个数据的下载链接
* config.json包括各个模式的设置及相应的batch_size

### 思路

bert-ner主要依赖BERT对输入的句子进行encode，然后经过CRF层对输出标签的顺序进行限制，对每个token预测一个分类标签，"B-PER"、"I-PER"、"O"等。

### 相比run_classifier.py的修改

* Features中的label_ids变为多个标签，即长度为N的文本，对应(N+2)个label([CLS], [SEP]).
* create_model变为get_sequence_output, 接入CRF模型
* 生成label_ids时的补充策略：定义`_X`标签作为一个特殊label，即不参与实体识别。通常应用在word piece和[CLS]、[SEP].

### 其他事项

* estimator.predict()返回max_seq_length个lable_id
* NERFastPredictor，免去每次预测重新加载模型
* 在预测出来结果后，text中每个词与predict_id一一对应，略过`_X`标签即可
* 标签以`B-`, `I-`，[`E-`]为标准
* nvidia-docker


### Docker-Nvidia

* docker18.03以下使用docker-nvidia1.0，以上使用docker-nvidia2

##### 17.12.1-ce正常运行

* 安装nvidia-docker1.0
```
yum install -y nvidia-docker
```

* 列出所有nvidia-docker volume
    - 应该有一个
    ```
    nvidia-docker volume ls
    ```
    * 如果没有的话
    ```shell
    docker volume create --driver=nvidia-docker --name=nvidia_driver_$(modinfo -F version nvidia)
    nvidia-docker volume ls
    ```
* 查看显卡和docker是否正常
    - 直接输出信息
    ```
    nvidia-docker run --rm nvidia/cuda nvidia-smi
    ```
    - 进入镜像查看
    ```sh
    # 直接查看失败的话进入镜像运行
    nvidia-docker run -it -p 8888:8888 --name ten tensorflow/tensorflow:0.11.0rc0-gpu  /bin/sh
    nvidia-smi
    python
    >>> import tensorflow as tf
    >>> tf.test.is_gpu_available()
    ```

* 构建容器
```
nvidia-docker build  --network host -t service:bert .
```

* 运行容器
```
nvidia-docker run --network host -it --rm -p 1234:1234 -v data:data  service:bert /bin/bash
```