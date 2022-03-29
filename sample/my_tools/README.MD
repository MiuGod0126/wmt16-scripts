## 角灰的文本处理工具

### 1.中文jieba**分词**，与多进程分词

```shell
python my_tools/cut.py src_file tgt_file

bash my_tools/cut.sh  workers src_file tgt_file
```

### 2.**词表转换**

json->vocab(paddle)->dict(fairseq)

```shell
# json转paddle vocab
python my_tools/json2vocab.py $lang $folder
# paddle vocab转fairseq dict
python my_tools/vocab2dict.py $lang $folder
fairseq dict 转  paddle voca
python my_tools/dict2vocab.py $lang $folder
```

### 3.**lang id过滤**(使用fasttext,需要下载模型文件)

```shell
python ./my_tools/data_filter.py --src-lang $SRC --tgt-lang $TRG --in-prefix data/train --out-prefix data/trainlang --threshold 0.5
# wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

### 4.批量**提取xml**和sgm

```shell
bash my_tools/process_xml_folder.sh infolder outfolders
```

### 5.合并src和tgt语料

```shell
python my_tools/merge.py src.txt tgt.txt outfile.txt
```

2022/3/21

### 6.流式、整句输入互转
如：
```
大
大家
大家晚         <=========> 大家晚上好
大家晚上
大家晚上好
```
