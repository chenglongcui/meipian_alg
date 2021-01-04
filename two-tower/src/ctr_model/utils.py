import tensorflow as tf


def get_item_cnt(recall_cnt_file):
    recall_file_list = tf.gfile.Glob(recall_cnt_file)
    with tf.gfile.Open(recall_file_list[0], 'r') as f:
        for line in f.readlines():
            line = line.strip().split(",")
        item_count = line[0]
        cate_cnt = line[1]
        tag_cnt = line[2]
    return int(item_count), int(cate_cnt), int(tag_cnt)


def get_ad_idx(ad_idx_file):
    recall_file_list = tf.gfile.Glob(ad_idx_file)
    with tf.gfile.Open(recall_file_list[0], 'r') as f:
        for line in f.readlines():
            line = line.replace('\"', "")
            ad_idx = [int(idx) for idx in line.split(",")]
    return ad_idx, len(ad_idx)


def get_file_name(file_path):
    file_list = tf.gfile.Glob(file_path)
    return file_list


def get_file_name_by_date(file_path):
    file_dict = {}
    file_list = tf.gfile.Glob(file_path)
    for file_name in file_list:
        file_name = file_name.split("/")
        date_str = file_name[-1].split("_")


def get_date_str(file_name):
    file_name = file_name.split("/")
    date_str = file_name[-1].split("_")

    # return file_list
