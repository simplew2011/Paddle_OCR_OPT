import os
import sys
import yaml

def parse_config(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    # 读取参数表
    with open(file_path, encoding='utf-8') as f:
        conf = yaml.safe_load(f)

    reader_file_path = conf['Global']['reader_yml']
    _, ext = os.path.splitext(reader_file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for reader"
    # 读取数据表
    with open(reader_file_path, encoding='utf-8') as f:
        reader_conf = yaml.safe_load(f)

    # 参数表和数据表合并，保存为新表
    conf.update(reader_conf)
    # with open('conf.yaml', 'w', encoding='utf-8') as f:
    #     yaml.dump(conf, f, sort_keys=False)
    return conf

if __name__ == '__main__':
    config_path = "./configs/rec/rec_yzm_train.yml"
    config_path1 = "./configs/rec/rec_yzm_reader.yml"

    with open(config_path, encoding='utf-8') as f:  # demo.yaml内容同上例yaml字符串
        conf = yaml.safe_load(f)
    conf['Global']['save_model_dir'] = './output/rec_CRNN222222'
    with open(config_path1, encoding='utf-8') as f:  # demo.yaml内容同上例yaml字符串
        conf1 = yaml.safe_load(f)

    conf.update(conf1)
    with open('demo5.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(conf, f, sort_keys=False)  # 写入文件，不是用flow流格式

    config = load_config(config_path)
    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    print(use_gpu)
    config['Global']['use_gpu'] = False
    config['Global']['save_model_dir'] = './output/rec_CRNN222222'
    print(use_gpu)
#python3 tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy


