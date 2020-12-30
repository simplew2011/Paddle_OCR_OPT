from tools import train

if __name__ == '__main__':

    #config_path = "./configs/rec/rec_yzm_train.yml"
    config_path = "./configs/det/det_mv3_db_v1.1_plate.yml"

    train.main(config_path)


# python tools/train.py -c configs/rec/rec_yzm_train.yml 2>&1 | tee train_rec1.log  18326104055
