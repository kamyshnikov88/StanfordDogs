import random

from pyflink.common import SimpleStringSchema, Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaProducer

from trainer import get_images_paths
from kafka import KafkaConsumer
from PIL import Image
import json
import time


SEP = b'S19283746E'
F = 150000


def get_f():
    return F


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')


def int_from_bytes(x_bytes: bytes) -> int:
    return int.from_bytes(x_bytes, 'big')


def get_extended_byte_string(img: Image):
    """
    Convert pil image to byte string and
     insert size of the image to begin of the string (in byte string too)
    :param img: pil image
    :return: byte string
    """
    w, h = img.size
    wb = int_to_bytes(w)
    hb = int_to_bytes(h)
    b_img_size = wb + SEP + hb + SEP
    b_size_with_image = b_img_size + img.tobytes()
    return b_size_with_image


def open_image(string: bytes) -> Image:
    if string == b'':
        return ''
    w, h, img = string.split(SEP)
    img = Image.frombytes('RGB',
                          (int_from_bytes(w), int_from_bytes(h)),
                          img)
    return img


def write_to_kafka_by_flink(values: list):
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    ds = env.from_collection(values, type_info=Types.STRING())
    kafka_sink = FlinkKafkaProducer(
        topic='img',
        serialization_schema=SimpleStringSchema(),
        producer_config={'bootstrap.servers': 'localhost:9092'
                         # 'max.in.flight.requests.per.connection': '1'
                         })
    ds.add_sink(kafka_sink)
    env.execute('flink_main')


def images_to_kafka_by_flink() -> None:
    n_msg = 0
    for _ in range(100):
        n = random.randint(1, 5)
        time.sleep(n)

        all_img_paths = get_images_paths()
        indices = [random.randint(0, len(all_img_paths)) for _ in range(n)]
        img_paths = all_img_paths[indices]
        to_flink = []
        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            ebs = get_extended_byte_string(img)
            bs = ebs.decode('latin1')
            lbs = len(bs)
            n_msg += lbs//F + 1
            if lbs > F:  # if kafka message can't contain whole image
                for j in range(lbs//F + 1):
                    sub_msg = bs[j*F:(j+1)*F]
                    value = {'i': i, 'j': j,
                             'time': int(round(time.time() * 1000)),
                             'json_value': sub_msg}
                    value = json.dumps(value)
                    to_flink.append(value)
            else:
                value = {'i': i, 'j': 0,
                         'time': int(round(time.time() * 1000)),
                         'json_value': bs}
                value = json.dumps(value)
                to_flink.append(value)

        # add technical value to close last window in flink while reading
        time.sleep(1.05)
        technical_value = {'i': -1, 'j': -1,
                           'time': int(round(time.time() * 1000)),
                           'json_value': ''}
        technical_value = json.dumps(technical_value)
        write_to_kafka_by_flink(to_flink + [technical_value])
        print('batch was delivered to kafka by flink')


def read_topic(topic_name: str) -> None:
    consumer = KafkaConsumer(topic_name,
                             bootstrap_servers='localhost:9092',
                             auto_offset_reset='earliest',
                             consumer_timeout_ms=1000)
    images = []
    msg = ''
    prev_i = 0
    prev_j = -1
    for i, item in enumerate(consumer):
        value = item.value
        if value == '':
            continue
        d = value.decode('latin1')
        jsn = json.loads(d)
        i, j, _, value = jsn.values()

        if i == prev_i and j > prev_j:
            msg += value
            prev_j += 1
        else:
            prev_i = i
            prev_j = -1
            images.append(open_image(msg.encode('latin1')))
            msg = value
    if msg != b'':
        images.append(open_image(msg.encode('latin1')))
    print(len([x for x in images if x != '']))


if __name__ == '__main__':
    images_to_kafka()
    images_to_kafka_by_flink()
    # read_topic('img')
