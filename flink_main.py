import json
import sys
from PIL import Image
import requests
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment, SourceFunction
from pyflink.datastream.connectors import (FlinkKafkaConsumer,
                                           FlinkKafkaProducer)
from pyflink.common.serialization import SimpleStringSchema
from kafka_client import open_image
from tfserving_prediction_getter import get_d
from tfserving_prediction_getter import get_url
import socket

socket.setdefaulttimeout(100)


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
    return d[pred.index(max(pred))]


def msg_to_pred(msg_list: list) -> list:
    images = []
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
                images.append(get_pred(open_image(msg).resize((299, 299))))
                msg = value
        else:
            prev_i += 1
            images.append(get_pred(open_image(msg).resize((299, 299))))
    if msg != b'':
        images.append(get_pred(open_image(msg).resize((299, 299))))
    return images


def flink_main() -> None:
    env = StreamExecutionEnvironment.get_execution_environment()
    # kafka_source = FlinkKafkaConsumer(
    #     topics='img3',
    #     deserialization_schema=SimpleStringSchema(),
    #     properties={'bootstrap.servers': 'localhost:9092'})
    # ds = env.add_source(kafka_source)

    ds = env.from_collection([b'123'], type_info=Types.BYTE())
    msg_list = []
    with ds.execute_and_collect() as results:
        for msg in results:
            msg_list.append(msg)
    predictions = msg_to_pred(msg_list)

    ds = env.from_collection(
        collection=predictions,
        type_info=Types.STRING())

    kafka_sink = FlinkKafkaProducer(
        topic='predictions',
        serialization_schema=SimpleStringSchema(),
        producer_config={'bootstrap.servers': 'localhost:9092'})
    ds.add_sink(kafka_sink)
    env.execute('flink_main')


if __name__ == '__main__':
    flink_main()



