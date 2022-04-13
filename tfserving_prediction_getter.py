from PIL import Image
import requests
import json


IMAGE_PATH = './data/Images/n02085620-Chihuahua/n02085620_10074.jpg'
MAPPING_DICT = {0: 'Afghan_hound', 1: 'African_hunting_dog', 2: 'Airedale', 3: 'American_Staffordshire_terrier', 4: 'Appenzeller', 5: 'Australian_terrier', 6: 'Bedlington_terrier', 7: 'Bernese_mountain_dog', 8: 'Blenheim_spaniel', 9: 'Border_collie', 10: 'Border_terrier', 11: 'Boston_bull', 12: 'Bouvier_des_Flandres', 13: 'Brabancon_griffon', 14: 'Brittany_spaniel', 15: 'Cardigan', 16: 'Chesapeake_Bay_retriever', 17: 'Chihuahua', 18: 'Dandie_Dinmont', 19: 'Doberman', 20: 'English_foxhound', 21: 'English_setter', 22: 'English_springer', 23: 'EntleBucher', 24: 'Eskimo_dog', 25: 'French_bulldog', 26: 'German_shepherd', 27: 'Gordon_setter', 28: 'Great_Dane', 29: 'Great_Pyrenees', 30: 'Greater_Swiss_Mountain_dog', 31: 'Ibizan_hound', 32: 'Irish_setter', 33: 'Irish_terrier', 34: 'Irish_water_spaniel', 35: 'Irish_wolfhound', 36: 'Italian_greyhound', 37: 'Japanese_spaniel', 38: 'Kerry_blue_terrier', 39: 'Labrador_retriever', 40: 'Lakeland_terrier', 41: 'Leonberg', 42: 'Lhasa', 43: 'Maltese_dog', 44: 'Mexican_hairless', 45: 'Newfoundland', 46: 'Norfolk_terrier', 47: 'Norwegian_elkhound', 48: 'Norwich_terrier', 49: 'Old_English_sheepdog', 50: 'Pekinese', 51: 'Pembroke', 52: 'Pomeranian', 53: 'Rhodesian_ridgeback', 54: 'Rottweiler', 55: 'Saint_Bernard', 56: 'Saluki', 57: 'Samoyed', 58: 'Scotch_terrier', 59: 'Scottish_deerhound', 60: 'Sealyham_terrier', 61: 'Shetland_sheepdog', 62: 'Siberian_husky', 63: 'Staffordshire_bullterrier', 64: 'Sussex_spaniel', 65: 'Tibetan_mastiff', 66: 'Tibetan_terrier', 67: 'Tzu', 68: 'Walker_hound', 69: 'Weimaraner', 70: 'Welsh_springer_spaniel', 71: 'West_Highland_white_terrier', 72: 'Yorkshire_terrier', 73: 'affenpinscher', 74: 'basenji', 75: 'basset', 76: 'beagle', 77: 'bloodhound', 78: 'bluetick', 79: 'borzoi', 80: 'boxer', 81: 'briard', 82: 'bull_mastiff', 83: 'cairn', 84: 'chow', 85: 'clumber', 86: 'coated_retriever', 87: 'coated_wheaten_terrier', 88: 'cocker_spaniel', 89: 'collie', 90: 'dhole', 91: 'dingo', 92: 'giant_schnauzer', 93: 'golden_retriever', 94: 'groenendael', 95: 'haired_fox_terrier', 96: 'haired_pointer', 97: 'keeshond', 98: 'kelpie', 99: 'komondor', 100: 'kuvasz', 101: 'malamute', 102: 'malinois', 103: 'miniature_pinscher', 104: 'miniature_poodle', 105: 'miniature_schnauzer', 106: 'otterhound', 107: 'papillon', 108: 'pug', 109: 'redbone', 110: 'schipperke', 111: 'silky_terrier', 112: 'standard_poodle', 113: 'standard_schnauzer', 114: 'tan_coonhound', 115: 'toy_poodle', 116: 'toy_terrier', 117: 'vizsla', 118: 'whippet'}
SERVER_URL = 'http://127.0.0.1:8501/v1/models/StanfordDogs_tf:predict'


def get_d() -> dict:
    return MAPPING_DICT


def get_url() -> str:
    return SERVER_URL


def get_prediction(path: str, d: dict) -> None:
    img = Image.open(path).resize((299, 299))
    instances = []
    for ch in zip(*img.getdata()):
        instances.append(list(map(lambda i: ch[i*299:(i+1)*299], range(299))))
    request = json.dumps({
        'instances': [instances]
    })

    response = requests.post(SERVER_URL, data=request)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise Exception(response.json()['error'])
    pred = response.json()['predictions'][0]
    print(f'Prediciton: {d[pred.index(max(pred))]}')


if __name__ == '__main__':
    get_prediction(IMAGE_PATH, MAPPING_DICT)
