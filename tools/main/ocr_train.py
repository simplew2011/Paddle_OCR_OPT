import os
import time
from configs import dxm_config
from tools import train, eval, export_model
import tools.utils.utils as ocr_utils
from ppocr.utils.logging import get_logger

logger = get_logger()


def ocr_train(conf_path, det_flag):

    # parse config from ocr_config.conf
    ocr_args = dxm_config(conf_path)
    ocr_args = ocr_utils.config_check(ocr_args)

    # judge ocr_args.dataset_dir is label_studio dataset or not, if checked, converter to ocr datasets
    dataset_dir, txt_path = ocr_utils.converter_from_label_studio(ocr_args.dataset_dir)
    if dataset_dir is not None and txt_path is not None:
        ocr_args.dataset_dir, ocr_args.label_file_path = dataset_dir, txt_path

    # prepare dataset
    if not det_flag:
        logger.info("rec train dataset extract form det dataset, it may take a few minutes, please waiting...")
        ocr_args.dataset_dir, ocr_args.label_file_path = ocr_utils.rec_dataset_preprocess(ocr_args.dataset_dir, ocr_args.label_file_path)
    split_ratio = 0.2
    train_txt_path, train_num, val_txt_path, val_num = ocr_utils.split_txt(ocr_args.label_file_path, split_ratio)
    logger.info("train dataset split (with {}) in train_num: {}, val_num: {}".format(split_ratio, train_num, val_num))

    if train_txt_path is None:
        # dataset invalid
        logger.error("error, please check out dataset: {}".format(dataset_dir))
        train_dict = {
            "flag": "error",
            "eval_dict": None,
            "inference_model_dir": None,
            "conf_path": None,
            "elapse": None
        }
        return train_dict

    # get default parameter
    args = ocr_utils.default_parse_config(det_train=det_flag, char_type=ocr_args.rec_char_type)

    # update args
    if det_flag:
        args["Global"]["epoch_num"] = ocr_args.det_epochs
        args["Global"]["save_model_dir"] = os.path.join(ocr_args.weight_save_dir, "det")
        args["Global"]["checkpoints"] = ocr_args.det_checkpoints_weight_dir
        args["Train"]["dataset"]["transforms"][3]["EastRandomCropData"]["size"] = [ocr_args.det_input_height, ocr_args.det_input_width]
        args["Train"]["loader"]["batch_size_per_card"] = ocr_args.det_batch_size
    else:
        args["Global"]["epoch_num"] = ocr_args.rec_epochs
        args["Global"]["save_model_dir"] = os.path.join(ocr_args.weight_save_dir, "rec")
        args["Global"]["checkpoints"] = ocr_args.rec_checkpoints_weight_dir
        args["Global"]["character_dict_path"] = ocr_args.rec_char_dict_path
        args["Global"]["character_type"] = ocr_args.rec_char_type
        args["Train"]["dataset"]["transforms"][3]["RecResizeImg"]["image_shape"] = [3, ocr_args.rec_input_height, ocr_args.rec_input_width]
        args["Eval"]["dataset"]["transforms"][2]["RecResizeImg"]["image_shape"] = [3, ocr_args.rec_input_height, ocr_args.rec_input_width]
        args["Train"]["loader"]["batch_size_per_card"] = ocr_args.rec_batch_size
        args["Eval"]["loader"]["batch_size_per_card"] = ocr_args.rec_batch_size

    args["Train"]["dataset"]["data_dir"] = ocr_args.dataset_dir
    args["Train"]["dataset"]["label_file_list"] = [train_txt_path]
    args["Eval"]["dataset"]["data_dir"] = ocr_args.dataset_dir
    args["Eval"]["dataset"]["label_file_list"] = [val_txt_path]

    try:
        start_time = time.time()

        # train
        logger.info("train processing...")
        train.main_new(args)

        # evaluation
        args["Global"]["checkpoints"] = os.path.join(args["Global"]["save_model_dir"], "best_accuracy")
        if not os.path.exists(args["Global"]["checkpoints"] + '.pdopt'):
            # no eval during training
            args["Global"]["checkpoints"] = os.path.join(args["Global"]["save_model_dir"], "latest")

        logger.info("eval processing...")
        eval_dict = eval.main_new(args)

        # convert to inference
        args["Global"]["pretrained_model"] = args["Global"]["checkpoints"]
        args["Global"]["save_inference_dir"] = os.path.join(os.path.split(args["Global"]["checkpoints"])[0], "inference")
        args["Global"]["load_static_weights"] = False
        logger.info("export processing...")
        export_model.main_new(args)

        train_dict = {
            "flag": "success",
            "eval_dict": eval_dict,
            "inference_model_dir": args["Global"]["save_inference_dir"],
            "conf_path": os.path.join(args["Global"]["save_model_dir"], "config.yml"),
            "elapse": time.time() - start_time
        }
        logger.info("success, train finished")
        return train_dict
    except Exception as err:
        logger.error("error, exception with: {}".format(err))
        train_dict = {
            "flag": "error",
            "eval_dict": None,
            "inference_model_dir": None,
            "conf_path": None,
            "elapse": None
        }
        return train_dict
