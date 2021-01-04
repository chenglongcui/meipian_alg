# encoding:utf-8
from bert_serving.client import BertClient
# from eas_prediction import PredictClient
# from eas_prediction import StringRequest
from pai_tf_predict_proto import tf_predict_pb2
import requests
import json
import time
import numpy as np


# bc = BertClient()


# # 由于延时高（1s）,放弃使用
# def get_bert_vector_eas(sentence_str):
#     model_name = 'bert_vector'
#     url = 'http://1247426312861996.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/bert_vector'
#     token = 'ODcxZjk3MzdhZTNiYjM0ZGEzNjk1ZWZjMjE1MWFiNDg3NzRiYzZjZQ=='
#
#     client = PredictClient(url, model_name)
#     client.set_token(token)
#     client.init()
#     # string_request = StringRequest(str(request_dict))
#     string = '{"first_sequence": "%s", "second_sequence": "", "sequence_length": 128, "output_schema": "pool_output"}' % sentence_str
#     string_request = StringRequest(string)
#     # 请求服务
#     resp = client.predict(string_request)
#     result = json.loads(resp.response_data)
#     print(result["pool_output"])
#
#     # string_2 = '{"first_sequence": "双11花呗花在哪里了", "second_sequence": "", "sequence_length": 128, "output_schema": "pool_output"}'
#     # string_request_2 = StringRequest(string_2)
#     # # 请求服务
#     # resp_2 = client.predict(string_request_2)
#     # print(resp_2)
#     # print string_1 == string_2


# def get_bert_vector(sentense_list):
#     # print(sentense_list)
#     bert_vector = bc.encode(sentense_list)
#     return bert_vector

def build_input(pai_request, input_name, dtype, batch_size, input):
    pai_request.inputs[input_name].dtype = dtype
    pai_request.inputs[input_name].array_shape.dim.extend([batch_size])
    pai_request.inputs[input_name].float_val.extend(input)


def topk_collection(sentense_str):
    # 输入模型信息,点击模型名字就可以获取到了

    url = 'http://1247426312861996.cn-beijing.pai-eas.aliyuncs.com/api/predict/tt_ctr'
    headers = {"Authorization": 'NzMwZjY4MzdhYWY0YWM4ZWYyODAzMGYwOTBkMDU3MzhjMDk4YWY1Yg=='}

    # 构造服务
    pai_request = tf_predict_pb2.PredictRequest()
    pai_request.signature_name = 'serving_default'

    # item_embedding = get_bert_vector(sentense_str)
    # print(item_embedding)

    batch_size = 1
    # print(batch_size)
    # 参数类型
    prediction_score = None
    user_id = [b'68947131']
    item_id = [b'256732189']
    class_id = [b'5']
    tag_ids = [b'124;']
    dev_type = [b'1']
    dev_brand = [b'oppo']
    dev_brand_type = [b'3']
    dev_os = [b'oppo_10']
    dev_net = [b'4']
    client_type = [b'0']
    dev_carrier = [b'1']
    click_seq_50size = [b';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;']
    gender = [b'1']
    age = [b'2']
    consume_level = [b'1']
    unclick_seq_50size = [b';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;']
    feed_type = [b'1004']

    pai_request.inputs['user_id'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['user_id'].array_shape.dim.extend([batch_size])
    pai_request.inputs['user_id'].string_val.extend(user_id)

    pai_request.inputs['item_id'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['item_id'].array_shape.dim.extend([batch_size])
    pai_request.inputs['item_id'].string_val.extend(item_id)

    pai_request.inputs['class_id'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['class_id'].array_shape.dim.extend([batch_size])
    pai_request.inputs['class_id'].string_val.extend(class_id)

    pai_request.inputs['tag_ids'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['tag_ids'].array_shape.dim.extend([batch_size])
    pai_request.inputs['tag_ids'].string_val.extend(tag_ids)

    pai_request.inputs['dev_type'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['dev_type'].array_shape.dim.extend([batch_size])
    pai_request.inputs['dev_type'].string_val.extend(dev_type)

    pai_request.inputs['dev_brand'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['dev_brand'].array_shape.dim.extend([batch_size])
    pai_request.inputs['dev_brand'].string_val.extend(dev_brand)

    pai_request.inputs['dev_brand_type'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['dev_brand_type'].array_shape.dim.extend([batch_size])
    pai_request.inputs['dev_brand_type'].string_val.extend(dev_brand_type)

    pai_request.inputs['dev_os'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['dev_os'].array_shape.dim.extend([batch_size])
    pai_request.inputs['dev_os'].string_val.extend(dev_os)

    pai_request.inputs['dev_net'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['dev_net'].array_shape.dim.extend([batch_size])
    pai_request.inputs['dev_net'].string_val.extend(dev_net)

    pai_request.inputs['client_type'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['client_type'].array_shape.dim.extend([batch_size])
    pai_request.inputs['client_type'].string_val.extend(client_type)

    pai_request.inputs['dev_carrier'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['dev_carrier'].array_shape.dim.extend([batch_size])
    pai_request.inputs['dev_carrier'].string_val.extend(dev_carrier)

    pai_request.inputs['click_seq_50size'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['click_seq_50size'].array_shape.dim.extend([batch_size])
    pai_request.inputs['click_seq_50size'].string_val.extend(click_seq_50size)

    pai_request.inputs['gender'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['gender'].array_shape.dim.extend([batch_size])
    pai_request.inputs['gender'].string_val.extend(gender)

    pai_request.inputs['age'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['age'].array_shape.dim.extend([batch_size])
    pai_request.inputs['age'].string_val.extend(age)

    pai_request.inputs['consume_level'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['consume_level'].array_shape.dim.extend([batch_size])
    pai_request.inputs['consume_level'].string_val.extend(consume_level)

    pai_request.inputs['unclick_seq_50size'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['unclick_seq_50size'].array_shape.dim.extend([batch_size])
    pai_request.inputs['unclick_seq_50size'].string_val.extend(unclick_seq_50size)

    pai_request.inputs['feed_type'].dtype = tf_predict_pb2.DT_STRING
    pai_request.inputs['feed_type'].array_shape.dim.extend([batch_size])
    pai_request.inputs['feed_type'].string_val.extend(feed_type)

    # # 将ProtoBuf序列化成string进行传输。
    pai_request_data = pai_request.SerializeToString()
    # print request_data
    resp = requests.post(url, data=pai_request_data, headers=headers)

    if resp.status_code != 200:
        print('Http status code: ', resp.status_code)
        print('Error msg in header: ', resp.headers)
        print('Error msg in body: ', resp.content)

    else:
        response = tf_predict_pb2.PredictResponse()
        response.ParseFromString(resp.content)
        prediction_score = response.outputs["prediction_score"].float_val
        print(prediction_score)
    # print(topk_score)
    # result_topk_collection = np.array(result_topk_collection).reshape((batch_size, -1))
    # result_topk_score = np.array(result_topk_score).reshape((batch_size, -1))
    return prediction_score


if __name__ == '__main__':
    # bert_vector = get_bert_vector("双11花呗花在哪里了")
    # print(bert_vector)
    sentense = "双11花呗花在哪里了"
    # t0 = time.time()
    topk_collections = topk_collection(sentense)
