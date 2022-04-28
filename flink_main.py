import functools
import json

from PIL import Image
import requests
from pyflink.common import Types, Time, WatermarkStrategy
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import (FlinkKafkaConsumer,
                                           FlinkKafkaProducer)
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.window import EventTimeSessionWindows
from kafka_client import open_image, get_f
from tfserving_prediction_getter import get_d
from tfserving_prediction_getter import get_url

MSG_SEP = '|msg||msg|'
SUB_MSG_SEP = '|sub_msg||sub_msg|'


class MyTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, value, record_timestamp) -> int:
        jsn = json.loads(value)
        _, _, time, _ = jsn.values()
        return time


def decode_key(msg: str) -> tuple:
    jsn = json.loads(msg)
    i, j, _, value = jsn.values()
    return i, str(j) + SUB_MSG_SEP + value if value != '' else ''


def fetch_parts(cum_msg: tuple, msg: tuple) -> tuple:
    v = cum_msg[-1] if MSG_SEP in cum_msg[-1] else cum_msg[-1] + MSG_SEP
    return cum_msg[0], v + msg[-1] + MSG_SEP


def get_predict(msg: tuple) -> str:
    if msg[1] == '':
        print('predict from flink was delivered to kafka (technical)')
        return ''
    img = open_image(sum_parts(msg).encode('latin1')).resize((299, 299))
    print('predict from flink was delivered to kafka')
    return get_pred(img)


def sum_parts(msg: tuple) -> str:
    value = msg[1][:-len(MSG_SEP)] if len(msg[1]) > get_f() else msg[1]
    if value == '':
        return ''
    if MSG_SEP in value:
        t = value.split(MSG_SEP)
    else:
        return value.split(SUB_MSG_SEP)[1]
    msg_j_v = list(map(lambda x: (x.split(SUB_MSG_SEP)[0],
                                  x.split(SUB_MSG_SEP)[1]),
                       t))
    msg = sorted(msg_j_v, key=lambda x: x[0])
    _, parts = zip(*msg)
    return functools.reduce(lambda x, y: x + y, parts)


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


def flink_main() -> None:
    env = StreamExecutionEnvironment.get_execution_environment()

    env.set_parallelism(1)

    kafka_source = FlinkKafkaConsumer(
        topics='img',
        deserialization_schema=SimpleStringSchema(),
        properties={'bootstrap.servers': 'localhost:9092',
                    'auto.offset.reset': 'earliest',
                    })
    ds = env.add_source(kafka_source)

    watermark_strategy = (WatermarkStrategy.
                          for_monotonous_timestamps()
                          .with_timestamp_assigner(MyTimestampAssigner()))

    ds = (ds
          .assign_timestamps_and_watermarks(watermark_strategy)
          .map(decode_key,
               output_type=Types.TUPLE([Types.INT(), Types.STRING()]))
          .key_by(lambda k: k[0], key_type=Types.INT())
          .window(EventTimeSessionWindows.with_gap(Time.seconds(1)))
          .reduce(reduce_function=lambda x, y: fetch_parts(x, y),
                  output_type=Types.STRING())
          .map(lambda e: get_predict(e), output_type=Types.STRING())
          )

    kafka_sink = FlinkKafkaProducer(
        topic='predictions',
        serialization_schema=SimpleStringSchema(),
        producer_config={'bootstrap.servers': 'localhost:9092'})
    ds.add_sink(kafka_sink)
    env.execute('flink_main')


if __name__ == '__main__':
    flink_main()



