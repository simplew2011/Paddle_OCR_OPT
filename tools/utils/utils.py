import os
import sys
import cv2
import yaml
import numpy as np
import glob
import shutil
from tqdm import tqdm
import simdjson
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from ppocr.utils.logging import get_logger

logger = get_logger()

project_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(project_dir, '../..'))


def get_boundrect(points):
    """
    获取点集所对应的外接距
    """
    assert len(points) == 4, "make sure annotation transcription in 4 points"
    x_min = min(points[0][0], points[3][0])
    y_min = min(points[0][1], points[1][1])
    x_max = max(points[1][0], points[2][0])
    y_max = max(points[2][1], points[3][1])
    return [[x_min, y_min], [x_max, y_max]]


def set_dir_empty(path_dir):
    """
    删除path路径下的所有文件及文件夹
    :param path_dir: 文件夹路径名 str
    :return: NONE
    """
    for i in os.listdir(path_dir):
        file_data = path_dir + "/" + i
        if os.path.isfile(file_data):
            os.remove(file_data)
        elif os.path.isdir(file_data):
            shutil.rmtree(file_data)


def find_file_by_suffix(path_dir, suffix_list=None):
    """
    基于给定文件路径和后缀列表，在该文件夹下查找所有后缀在列表内的文件
    :param path_dir: 文件夹路径
    :param suffix_list: 给定参考文件后缀列表，如["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
    :return files_list: 在文件夹内符合后缀的文件路径组成的列表
    """
    if suffix_list is None:
        suffix_list = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.gif", "*.pgm"]
    files_list = []
    for suffix in suffix_list:
        files_list.extend(glob.glob(os.path.join(path_dir, suffix)))
    return files_list


def find_image_path_by_ann(ann_file_name, dataset_dir):
    """
    由json文件名在数据集文件夹内找对应的图片路径（可适配两种 [1.jpg, 1.json] [1.jpg, 1.jpg.json]）
    :param ann_file_name: 标注文件的文件名
    :param dataset_dir: 文件夹路径
    :return image_path: 对应图片路径
    """
    image_path = None
    suffix_list = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.gif", "*.pgm"]

    if os.path.isdir(dataset_dir):
        images = find_file_by_suffix(dataset_dir, suffix_list)
        for suffix in suffix_list:
            image_name = ann_file_name.replace(os.path.splitext(ann_file_name)[-1], suffix[1:])  # 1.jpg 1.json
            image_add_name = os.path.splitext(ann_file_name)[0]     # 1.jpg 1.jpg.json
            if os.path.join(dataset_dir, image_name) in images:
                image_path = os.path.join(dataset_dir, image_name)
                return image_path
            elif os.path.join(dataset_dir, image_add_name) in images:
                image_path = os.path.join(dataset_dir, image_add_name)
                return image_path
    return image_path


def check_ocr_xml(xml_path):
    """
    判断xml文件是否为ocr任务
    :param xml_path: xml文件路径
    :return bool:
    """
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    object_set = root.findall("OcrLabels")
    if object_set:
        return True
    else:
        logger.error("{} is not a valid label_studio xml file.".format(xml_path))
        return False


def label_studio_to_ocr(json_path):
    """
    将label_studio的JSON数据转为ocr数据结构
    :param json_path: label_studio的标注JSON文件路径
    :return annotation_dict: JSON内容对应的ocr数据字典
    """
    dataset_dir = os.path.dirname(json_path)
    image_path = find_image_path_by_ann(os.path.basename(json_path), dataset_dir)
    if image_path is None:
        logger.error("failed to find an image file corresponding to JSON: {}".format(json_path))
        return {}

    records = []
    with open(json_path, 'rb') as fin:
        records = simdjson.loads(fin.read())

    if len(records) < 1:
        logger.error("{} is not a valid label_studio json file.".format(json_path))
        return {}
    each_dic = records[0]['completions'][0]['result']

    box_points = []
    for num in range(len(each_dic)):
        value = each_dic[num]['value']
        label_dict = {
            "transcription": value['ocrlabels'][0],
            "points": [[int(x), int(y)] for x, y in value['points']]}
        box_points.append(label_dict)

    annotation_dict = {
        "image_path": os.path.basename(image_path),
        "annotation": box_points
    }
    return annotation_dict


def converter_from_label_studio(dataset_dir):
    """
    判断数据集路径是否为label_studio格式，如是，则转换为ocr专用格式
    """
    if not os.path.isdir(dataset_dir):
        logger.error("{} is not a valid dataset dir path.".format(dataset_dir))
        return None, None
    suffix_list = ["*.xml"]
    conf_list = find_file_by_suffix(dataset_dir, suffix_list)
    if not conf_list:
        return None, None
    if not check_ocr_xml(conf_list[0]):
        return None, None

    logger.info("label_studio dataset convert to ocr dataset cycling, it may take a few minutes, please waiting...")

    images_list = find_file_by_suffix(dataset_dir)

    annotations = []
    for image_path in tqdm(images_list):
        if os.path.exists(image_path + ".json"):
            json_path = image_path + ".json"
        elif os.path.exists(os.path.splitext(image_path)[0] + ".json"):
            json_path = os.path.splitext(image_path)[0] + ".json"
        else:
            logger.warning("The JSON file corresponding to the image: {} could not be found".format(image_path))
            continue

        annotation_dict = label_studio_to_ocr(json_path)
        if annotation_dict:
            annotations.append(annotation_dict)

    txt_path = os.path.join(dataset_dir, "annotations.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for ann in annotations:
            if not ann["annotation"]:
                continue
            f.write(ann["image_path"] + '\t' + simdjson.dumps(ann["annotation"], ensure_ascii=False) + '\n')
    return dataset_dir, txt_path


def rec_dataset_preprocess(dataset_dir, txt_file_path):
    """
    从检测数据集中转换出识别数据集
    """
    with open(txt_file_path, encoding='utf-8') as f:
        lines = f.readlines()

    # 定义识别数据集图片文件夹及标注文本
    rec_txt_file_path = txt_file_path.replace(".txt", "_rec.txt")
    rec_image_dir = os.path.join(os.path.split(txt_file_path)[0], 'rec_dataset')

    if not os.path.exists(rec_image_dir):
        os.makedirs(rec_image_dir)

    # 解析检测标注文件，截取文字区域和获取对应文字标签
    with open(rec_txt_file_path, "w", encoding='utf-8') as fout:
        for line in tqdm(lines):
            substr = line.strip("\n").split("\t")
            image_path = os.path.join(dataset_dir, substr[0])
            image = cv2.imread(image_path)
            if image is None:
                print("{} is not a valid image path.".format(image_path))
                continue

            label = simdjson.loads(substr[1])
            nBox = len(label)
            for bno in range(0, nBox):
                points = label[bno]['points']
                box = get_boundrect(points)
                txt = label[bno]['transcription']

                crop_image = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]

                one_image_path = os.path.splitext(substr[0])[0] + "_" + str(bno) + os.path.splitext(substr[0])[-1]
                crop_image_path = os.path.join(rec_image_dir, one_image_path)
                cv2.imwrite(crop_image_path, crop_image)

                txt_str = one_image_path + "\t" + txt + "\n"
                fout.write(txt_str)

    return rec_image_dir, rec_txt_file_path


def default_parse_config(config_path=None, det_train=True, char_type="ch"):
    """
    解析配置信息，如未提供配置表路径，则使用默认配置表
    """
    if config_path is not None:
        file_path = config_path
    else:
        # 加载默认检测或识别配置参数
        if det_train:
            file_path = os.path.join(project_dir, "./configs/det/det_mv3_db.yml")
        else:
            if char_type == "ch":
                file_path = os.path.join(project_dir, "./configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml")
            else:
                file_path = os.path.join(project_dir, "./configs/rec/multi_language/rec_en_number_lite_train.yml")

    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    # 读取参数表
    with open(file_path, encoding='utf-8') as f:
        conf = yaml.safe_load(f)

    # fix path error, when this project set as third_part module
    conf["Global"]["pretrained_model"] = os.path.join(project_dir, conf["Global"]["pretrained_model"])

    return conf


def split_txt(txt_file_path, ratio=0.2):
    """
    按比例拆分训练数据
    """
    if not os.path.exists(txt_file_path):
        logger.error('train dataset txt file none exists in {}'.format(txt_file_path))
        return None, None, None, None

    with open(txt_file_path, encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 10:
        logger.error('the valid annotation less then 10')
        return None, None, None, None
    elif lines[-1][-1] != '\n':
        lines[-1] = lines[-1] + '\n'

    np.random.seed(10101)
    np.random.shuffle(lines)
    num_val = int(len(lines) * ratio)
    num_train = len(lines) - num_val
    train_txt = txt_file_path.replace(".txt", "_train.txt")
    val_txt = txt_file_path.replace(".txt", "_val.txt")

    with open(train_txt, "w", encoding='utf-8') as f:
        for line in lines[num_val:]:
            f.write(line)
    with open(val_txt, "w", encoding='utf-8') as f:
        for line in lines[:num_val]:
            f.write(line)
    return train_txt, num_train, val_txt, num_val


def config_check(config):
    """
    配置校验
    """
    if config.det_checkpoints_weight_dir == "":
        config.det_checkpoints_weight_dir = None
    if config.dataset_dir[-1] != '/' and config.dataset_dir[-1] != r'\\':
        config.dataset_dir = config.dataset_dir + '/'
    if config.det_model_dir == "":
        config.det_model_dir = os.path.join(project_dir, "./weights/det/ch_ppocr_server_v2.0_det_infer")
    if config.rec_model_dir == "":
        config.rec_model_dir = os.path.join(project_dir, "./weights/rec/ch_ppocr_server_v2.0_rec_infer")
        if config.rec_char_type == 'en':
            config.rec_model_dir = os.path.join(project_dir, "./weights/rec/en_number_mobile_v2.0_rec_infer")
            config.rec_char_dict_path = os.path.join(project_dir, "./ppocr/utils/dicts/en_dict.txt")
    if config.rec_char_dict_path == "":
        config.rec_char_dict_path = os.path.join(project_dir, "./ppocr/utils/dicts/ppocr_keys_v1.txt")
    return config


if __name__ == "__main__":
    # xml_path = r"C:\Users\Administrator\Desktop\test\label_studio_datasets\ocr\config.xml"
    # check_ocr_xml(xml_path)
    json_path = r"C:\Users\Administrator\Desktop\test\label_studio_datasets\ocr\1615259817828464103.jpg.json"
    label_studio_to_ocr(json_path)
    pass
