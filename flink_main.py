import functools
import json

from PIL import Image
import requests
from matplotlib import pyplot as plt
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import (FlinkKafkaConsumer,
                                           FlinkKafkaProducer)
from pyflink.common.serialization import SimpleStringSchema
from kafka_client import open_image
from tfserving_prediction_getter import get_d
from tfserving_prediction_getter import get_url


def get_pred(img: Image) -> str:
    d = get_d()
    server_url = get_url()
    instances = []
    for ch in zip(*img.getdata()):
        instances.append(list(map(lambda i: ch[i*299:(i+1)*299], range(299))))
    request = json.dumps({
        'instances': [instances]
    })

    response = requests.post(server_url, data=request)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise Exception(response.json()['error'])
    pred = response.json()['predictions'][0]
    print('get_pred FINALLY')
    return d[pred.index(max(pred))]


def msg_to_pred(msg_list: list) -> list:
    preds = []
    msg = b''
    prev_i = 0
    for msg in msg_list:
        key = msg.key.decode()
        value = msg.value
        if '|' in key:
            if int(key.split('|')[0]) == prev_i:
                msg += value
            else:
                prev_i = int(key.split('|')[0])
                preds.append(get_pred(open_image(msg).resize((299, 299))))
                msg = value
        else:
            prev_i += 1
            preds.append(get_pred(open_image(value).resize((299, 299))))
    if msg != b'':
        preds.append(get_pred(open_image(msg).resize((299, 299))))
    return preds


def decode_key(msg: str) -> tuple:
    jsn = json.loads(msg)
    i, j, value = jsn.values()
    return i, j, value


def fetch_parts(cum_msg, msg):
    print(123)
    cum_msg_v = cum_msg[-1]
    if isinstance(cum_msg_v, str):
        cum_msg_v = [tuple([cum_msg[1], cum_msg[-1]])]
    cum_msg_v.append(tuple([msg[1], msg[-1]]))
    return cum_msg[0], 0, cum_msg_v


def sum_parts(msg: list):
    # print(type(msg))
    # print(type(msg))
    # print(len(msg))
    # msg = sorted(msg, key=lambda x: x[0])
    # _, parts = zip(*msg)
    # return sum(parts)
    return 0


def flink_main() -> None:
    env = StreamExecutionEnvironment.get_execution_environment()
    project_path = 'file:///Users/kamyshnikovy/PycharmProjects/StanfordDogs/'
    flink_path = project_path + 'data/flink/'
    jar_names = [
        'flink-connector-kafka_2.12-1.14.4.jar',
        'kafka-clients-2.4.1.jar',
        'flink-connector-base-1.14.4.jar',
        'flink-shaded-force-shading-14.0.jar',
        'lz4-java-1.6.0.jar',
        'slf4j-api-1.7.28.jar',
        'snappy-java-1.1.7.3.jar',
        'zstd-jni-1.4.3-1.jar'
        ]
    jar_names = list(map(lambda x: flink_path + x, jar_names))

    env.add_jars(*jar_names)
    kafka_source = FlinkKafkaConsumer(
        topics='img',
        deserialization_schema=SimpleStringSchema(),
        properties={'bootstrap.servers': 'localhost:9092',
                    'auto.offset.reset': 'earliest'
                    })
    ds = env.add_source(kafka_source)

    ds = (ds
          .map(decode_key)
          .key_by(lambda x: x[0])
          .reduce(lambda c, d: fetch_parts(c, d))
          .map(lambda z: print(len(z[-1])))
          .map(sum_parts)
          .map(lambda x: get_pred(open_image(x.encode('latin1')).resize((299, 299)))))

    # kafka_sink = FlinkKafkaProducer(
    #     topic='predictions',
    #     serialization_schema=SimpleStringSchema(),
    #     producer_config={'bootstrap.servers': 'localhost:9092'})
    # ds.add_sink(kafka_sink)
    env.execute('flink_main')


if __name__ == '__main__':
    # flink_main()



