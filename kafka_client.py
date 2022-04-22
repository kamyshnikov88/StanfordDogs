from trainer import get_images_paths
from kafka import KafkaProducer, KafkaConsumer
from PIL import Image
import json
import time


SEP = b'S19283746E'


def get_separator():
    return SEP


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
    w, h, img = string.split(SEP)
    img = Image.frombytes('RGB',
                          (int_from_bytes(w), int_from_bytes(h)),
                          img)
    return img


def images_to_kafka() -> None:
    img_paths = get_images_paths()[:1]
    producer = KafkaProducer(bootstrap_servers='localhost:9092')
    count = 0
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        bs = get_extended_byte_string(img).decode('latin1')
        lbs = len(bs)
        f = 150000
        if lbs > f:  # if kafka message can't contain whole image
            for j in range(lbs//f + 1):
                count += 1
                sub_msg = bs[j*f:(j+1)*f]
                value = {'i': i, 'j': j,
                         'time': int(round(time.time() * 1000)),
                         'json_value': sub_msg}
                value = json.dumps(value).encode('latin1')
                producer.send('img', value=value)
        else:
            count += 1
            value = {'i': i, 'j': 0,
                     'time': int(round(time.time() * 1000)),
                     'json_value': bs}
            value = json.dumps(value).encode('latin1')
            producer.send('img', value=value)


def read_topic(topic_name: str) -> None:
    consumer = KafkaConsumer(topic_name,
                             bootstrap_servers='localhost:9092',
                             auto_offset_reset='earliest',
                             consumer_timeout_ms=1000)
    images = []
    msg = ''
    prev_i = 0
    for item in consumer:
        d = item.value.decode('latin1')
        jsn = json.loads(d)
        i, j, _, value = jsn.values()

        if i == prev_i:
            msg += value
        else:
            prev_i = i
            images.append(open_image(msg.encode('latin1')))
            msg = value
    if msg != b'':
        images.append(open_image(msg.encode('latin1')))


if __name__ == '__main__':
    images_to_kafka()
    read_topic('img')
