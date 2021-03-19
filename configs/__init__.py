from __future__ import absolute_import
import os
import io
import argparse
import glob
from collections import defaultdict
import configparser

__all__ = ["dxm_config", "ocr_config"]


def unique_config_sections(config_file):
    """
    加载配置参数时格式转义，防止编码格式异常
    Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file, encoding='utf-8') as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def dxm_config(conf_file_path):
    """
    根据路径解析配置表
    :param conf_file_path: 配置表文件路径
    :return: 配置表对象
    """
    conf_file_path = unique_config_sections(conf_file_path)
    config = configparser.ConfigParser()
    config.read_file(conf_file_path)
    secs = config.sections()
    args_init = argparse.ArgumentParser().parse_args()

    args_init.model_type = config.get(secs[0], 'model_type').strip()

    # 不同任务所需配置内容不一致
    if args_init.model_type == 'ocr':
        args = ocr_config(config, secs, args_init)
    else:
        assert args_init.model_type in ['ocr'], 'using model type is invalid, please confirm'

    return args


def ocr_config(config, secs, args):
    """
    分OCR任务的配置表解析（每种任务中配置项有差异）
    :param config:
    :param secs:
    :param args:
    :return:
    """
    args.model_type = config.get(secs[0], 'model_type').strip()
    args.model_name = 'ocr'
    args.weight_save_dir = config.get(secs[1], "weight_save_dir").strip()
    args.det_checkpoints_weight_dir = config.get(secs[1], "det_checkpoints_weight_dir").strip()
    args.rec_checkpoints_weight_dir = config.get(secs[1], "rec_checkpoints_weight_dir").strip()

    # dataset
    args.dataset_dir = config.get(secs[1], "dataset_dir").strip()
    args.label_file_path = config.get(secs[1], "label_file_path").strip()

    args.rec_char_dict_path = config.get(secs[1], "rec_char_dict_path").strip()
    args.rec_char_type = config.get(secs[1], "rec_char_type").strip()

    # train
    args.det_epochs = config.getint(secs[1], "det_epochs")
    args.det_batch_size = config.getint(secs[1], "det_batch_size")
    args.det_input_width = config.getint(secs[1], "det_input_width")
    args.det_input_height = config.getint(secs[1], "det_input_height")

    args.rec_epochs = config.getint(secs[1], "rec_epochs")
    args.rec_batch_size = config.getint(secs[1], "rec_batch_size")
    args.rec_input_width = config.getint(secs[1], "rec_input_width")
    args.rec_input_height = config.getint(secs[1], "rec_input_height")

    # test
    args.test_image_dir = config.get(secs[2], "test_image_dir").strip()
    args.test_result_dir = config.get(secs[2], "test_result_dir").strip()
    args.det_model_dir = config.get(secs[2], "det_model_dir").strip()
    args.rec_model_dir = config.get(secs[2], "rec_model_dir").strip()

    return args
