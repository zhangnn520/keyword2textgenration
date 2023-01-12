# coding=utf-8
import os
import re
import csv
import xlrd
import json
import copy
import random
import codecs
import time
import logging
import pandas as pd
from functools import reduce
from pandas import DataFrame
from tqdm import tqdm

base_dir = os.getcwd()


def read_text(path):
    with open(path, "r", encoding="utf-8") as reader:
        return [i.replace("\n", "") for i in reader.readlines()]


# 用户自定义词典

def find_chinese_word(string):
    string = string.strip()
    pattern = r"[\u4e00-\u9fa5]+"
    pat = re.compile(pattern)
    result = pat.findall(string)
    if result:
        return " ".join(result)
    else:
        return ""


def find_english_word(string):
    pattern = r"[a-zA-Z]+|（[a-zA-Z]+）"
    pat = re.compile(pattern)
    result = pat.findall(string.strip())
    return " ".join(result)


def find_index(string):
    pattern = r"[A-Z0-9,，]+"
    pat = re.compile(pattern)
    result_list = pat.findall(string)
    if result_list:
        return [result.replace(",", " ").replace("，", " ") for result in result_list][0]
    else:
        return ""


def write_csv(path, fieldnames, data_list):
    f = open(path, 'w+', encoding='utf-8-sig', newline="")
    csv_write = csv.writer(f)
    csv_write.writerow(fieldnames)
    for data in data_list:
        csv_write.writerow([data['id'], data['input_text'], data['target_text']])
    f.close()


def generate(arg, texts_list, tokenizer, model):
    """
    句子生成
    :param model:
    :param texts_list:
    :param tokenizer:
    :return: result
    """
    result_list = []
    texts = [i['input_text'].replace("# ", "") for i in texts_list]
    for num, txt in tqdm(enumerate(texts)):
        model.eval()
        temp_dict = copy.deepcopy(texts_list[num])
        temp_dict['input_text'] = temp_dict['input_text'].replace("# ", "")
        temp_dict['target_text'] = temp_dict['target_text'].replace("# ", "")
        input_ids = tokenizer.encode("keywords:{} </s>".format(txt), return_tensors="pt")
        outputs = model.generate(input_ids, max_length=arg.max_length, min_length=0)
        result = re.sub('<pad> |</s>|# ', "", tokenizer.decode(outputs[0], errors='ignore'))
        temp_dict.update({"predict_text": result})
        result_list.append(temp_dict)
    write_json(arg.predict_file_path, result_list)
    return result_list


def write_logger_info(args):  # 建立日志
    logger = logging.getLogger()  # 用getLogger()方法得到一个日志记录器
    logger.setLevel(logging.INFO)  # 设置成debug模式
    command_handler = logging.StreamHandler()  # 显示日志:用logging.StreamHandler()得到一个显示管理对象
    command_handler.setLevel(logging.INFO)
    logger.addHandler(command_handler)  # 调用记录器的addHandler()方法添加显式管理对象
    file_handler = logging.FileHandler('{}/{}.log'.format(args.logging_path, args.network))  # 在指定目录下保存train.log
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)  # 添加文件管理对象
    return logger.info(args)


class logFrame:
    def logg(self, arg):
        # 创建一个日志器
        logger = logging.getLogger('logger')
        # 判断处理器是否存在，如果有处理器就不添加，如果不存在处理器就添加处理器
        if not logger.handlers:
            # 设定日志器的日志级别（如果不设定，默认展示WARNING级别以上的日志）
            logger.setLevel(logging.DEBUG)
            # 创建一个处理器， StreamHandler() 控制台实现日志输出
            sh = logging.StreamHandler()
            # 创建处理器，FileHandler() 将日志输出到文件保存
            fh = logging.FileHandler(
                os.path.join(arg.logging_path, f'{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}_log.txt'),
                encoding='utf-8')
            # datefmt 表示输出日期的格式
            lf = logging.Formatter(fmt='%(asctime)s | %(filename)s：%(lineno)d line | %(levelname)s | %(message)s',
                                   datefmt='%Y_%m_%d %H:%M:%S')

            # 控制台输出日志
            # 在日志器中加入处理器
            logger.addHandler(sh)
            # 处理器中设置日志输出格式
            sh.setFormatter(lf)
            # 给处理器设置级别
            sh.setLevel(logging.INFO)

            # 文件保存日志
            logger.addHandler(fh)
            fh.setFormatter(lf)
        return logger


def get_string_index(string, string_one):
    index_num = -1
    index_list = []
    string_length = len(string_one)
    if string_length:
        b = string.count(string_one)
        for i in range(b):  # 查找所有的下标
            index_num = string.find(string_one, index_num + 1, len(string))
            index_list.append([str(index_num), str(index_num + string_length)])
    return index_list


def read_json(path):
    with open(path, "r", encoding="utf-8") as reader:
        return json.loads(reader.read())


def get_content(object_list, content_name="CONTENT"):
    return [content[f'{content_name}'] for content in object_list]


def write_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            writer.write(content + "\n")


def write_text3(write_path, content_lines):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content_line in content_lines:
            for line in content_line:
                writer.write(line + "\n")
            writer.write("\n")


def write_json_by_line(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for line in content_line:
            writer.write(json.dumps(line, ensure_ascii=False))
            writer.write("\n")


def write_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        writer.write(json.dumps(content_line, ensure_ascii=False, indent=4))


def get_maintain_clear_data(english_maintain, chinese_maintain):
    english_clear_data_list, chinses_clear_data_list = list(), list()
    for en in tqdm(english_maintain):
        for ch in chinese_maintain:
            if en['STATISTICS'] == ch['STATISTICS'] and en['HTML'].split("/")[-1] == ch['HTML'].split("/")[-1]:
                english_clear_data_list.append(en)
                chinses_clear_data_list.append(ch)
    try:
        assert len(english_clear_data_list) == len(chinses_clear_data_list)
    except Exception as e:
        print(e)
    return english_clear_data_list, chinses_clear_data_list


def fast_align_data(en_list, ch_list):
    en2ch_list = list()
    assert len(en_list) == len(ch_list)
    for num, ch_content in enumerate(ch_list):
        en_content = en_list[num]
        new_content = en_content + ' ||| ' + ch_content
        en2ch_list.append(new_content)
    return en2ch_list


def sub_word(string):
    pattern_one = u"\\(.*?\\)|\\（.*?）|\\[.*?]|-|“|”|``|''"
    result = re.sub(pattern_one, "", string)
    pattern_two = u" +"
    result = re.sub(pattern_two, " ", result)

    return result


def write_line_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for line in content_line:
            writer.write(json.dumps({line: content_line[line]}, ensure_ascii=False))
            writer.write("\n")


def merge_xls_sheet(excel_name):
    # 读取excel
    wb = xlrd.open_workbook(excel_name)
    sheets = wb.sheet_names()
    # 合并sheet
    all_data = DataFrame()
    temp_dict = dict()
    for i in range(len(sheets)):
        j = 1
        df = pd.read_excel(excel_name, sheet_name=i, header=None)
        all_data = all_data.append(df)
        j += 1
        temp_dict[sheets[i]] = df.values.tolist()
    # todo 目前已经完成中英文拆分，后续还需要图片名称和零件位置写入库中
    content_list = all_data[1].values.tolist()[1:]
    return content_list, all_data, temp_dict


def write_bio(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for contents in content_line:
            for content in contents:
                writer.write(content + "\n")


def write_bio2(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for contents in content_line:
            writer.writelines(contents)
            writer.write("\n")


def read_csv(read_path, data_split_rate, model="gbk"):
    content_list = list()
    with codecs.open(read_path, encoding=model) as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            temp_dict = dict()
            temp_dict['input_text'] = row['input_text']
            temp_dict['target_text'] = row['target_text']
            content_list.append(temp_dict)
    content_list = delete_duplicate_elements(content_list)
    list_length = len(content_list)
    random.shuffle(content_list)
    train_data = content_list[:int(list_length * data_split_rate)]
    eval_or_test_data = content_list[int(list_length * data_split_rate):]
    return train_data, eval_or_test_data, list_length


def delete_duplicate_elements(list_data):
    return reduce(lambda x, y: x if y in x else x + [y], [[], ] + list_data)


def write_ann_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            writer.write(content + "\n")
