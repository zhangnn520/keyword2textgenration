#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from tools.tools import read_text, write_csv

base_dir = os.path.dirname(__file__)


class DataProcessor(object):
    def __init__(self):
        self.bio_list = list()
        self.all_bio_list = list()
        self.csv_path = os.path.join(base_dir, "../demo_data.csv")
        self.head_list = ["id", "input_text", "target_text"]
        self.demo_file_path = os.path.join(base_dir, "demo_data.txt")

    def get_bio_list(self):
        ann_bio_list = read_text(self.demo_file_path)
        for bio_data in ann_bio_list:
            if bio_data:
                self.bio_list.append(bio_data)
            else:
                self.all_bio_list.append(self.bio_list)
                self.bio_list = list()
        return self.all_bio_list

    @staticmethod
    def get_ann_words(bio_data):
        """
        将标注数据标注的词语拿出来，并返回
        :param bio_data:
        :return keyword
        """
        entities, entity_tags, words, targets = list(), list(), list(), list()
        start, end, start_flag = 0, 0, False
        for line in bio_data:
            if line != "\n":
                words.append(line.split("\t")[0])
                targets.append(line.split("\t")[1])
        for idx, tag in enumerate(targets):
            if tag.startswith('B-'):  # 一个实体开头 另一个实体（I-）结束
                end = idx
                if start_flag:  # 另一个实体以I-结束，紧接着当前实体B-出现
                    entities.append("".join(words[start:end]))
                    entity_tags.append(targets[start][2:].lower())
                    start_flag = False
                start = idx
                start_flag = True
            elif tag.startswith('I-'):  # 实体中间，不是开头也不是结束，end+1即可
                end = idx
            elif tag.startswith('O'):  # 无实体，可能是上一个实体的结束
                end = idx
                if start_flag:  # 上一个实体结束
                    entities.append("".join(words[start:end]))
                    entity_tags.append(targets[start][2:].lower())
                    start_flag = False
        if start_flag:  # 句子以实体I-结束，未被添加
            entities.append("".join(words[start:end + 1]))
            entity_tags.append(targets[start][2:].lower())
            start_flag = False
        return entities, entity_tags

    @classmethod
    def get_bio_content(cls, bio_data):
        word_list = [data.split("\t")[0] for data in bio_data]
        return "".join(word_list)

    def __call__(self):

        content_keyword_list = list()
        ner_data = self.get_bio_list()
        for num, data in enumerate(ner_data):
            temp_dict = dict()
            temp_dict['id'] = num+1
            # temp_dict['prefix'] = "webNLG"
            entities_list, _ = DataProcessor.get_ann_words(data)
            temp_dict['input_text'] = " | ".join(entities_list)
            content = DataProcessor.get_bio_content(data)
            temp_dict['target_text'] = content
            if " | ".join(entities_list) and content:
                content_keyword_list.append(temp_dict)
        write_csv(self.csv_path, self.head_list, content_keyword_list)
        return content_keyword_list


if __name__ == "__main__":
    data_processor = DataProcessor()
    data_processor()
