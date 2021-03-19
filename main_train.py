from tools.main.ocr_train import ocr_train

if __name__ == "__main__":
    # 配置参数路径
    conf_file_path = "./configs/ocr_config.conf"
    # 训练任务标识，True为训练文字检测，False为训练文字识别
    det_train = True    # True  False
    # train_res_dict = ocr_train(conf_file_path, det_train)
    train_rec_dict = ocr_train(conf_file_path, not det_train)

    print("ocr train end with: {}".format(train_rec_dict))
