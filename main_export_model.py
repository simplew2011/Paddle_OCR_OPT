from tools import export_model

if __name__ == '__main__':
    config_path = "./configs/rec/rec_yzm_train.yml"
    config_path = "./configs/det/det_mv3_db_v1.1_plate.yml"

    export_model.main(config_path)



# python tools/export_model.py -c configs/rec/rec_yzm_train1.yml -o Global.checkpoints=output/rec_CRNN2/best_accuracy Global.save_inference_dir=output/rec_CRNN2/inference
