import os
import re
import html
import string
import torch
import config
import unicodedata
from nltk.tokenize import word_tokenize

from dataset import XRayDataset
from model import EncoderDecoderNet
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def load_dataset(raw_caption=False):
    return XRayDataset(
        root=config.DATASET_PATH,
        transform=config.basic_transforms,
        freq_threshold=config.VOCAB_THRESHOLD,
        raw_caption=raw_caption
    )


def get_model_instance(vocabulary):
    model = EncoderDecoderNet(
        features_size=config.FEATURES_SIZE,
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        vocabulary=vocabulary,
        encoder_checkpoint='./weights/chexnet.pth.tar'
    )
    model = model.to(config.DEVICE)

    return model

def train_test_split(dataset, test_size=0.25, random_state=44):
    train_idx, test_idx = sklearn_train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        random_state=random_state
    )

    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def save_checkpoint(checkpoint):
    print('=> Saving checkpoint')

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(config.CHECKPOINT_FILE), exist_ok=True)

    torch.save(checkpoint, config.CHECKPOINT_FILE)



def load_checkpoint(model, optimizer=None):
    print('=> Loading checkpoint')

    checkpoint = torch.load(config.CHECKPOINT_FILE, weights_only=False)  # Explicitly disable weights_only
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint.get('epoch', 0)  # Default to 0 if 'epoch' key is missing



def can_load_checkpoint():
    return os.path.exists(config.CHECKPOINT_FILE) and config.LOAD_MODEL


def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')

    return re1.sub(' ', html.unescape(x1))


def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    return re.sub(r'\d+', '', text)


def text2words(text):
    return word_tokenize(text)


def normalize_text( text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)

    return text


def normalize_corpus(corpus):
    return [normalize_text(t) for t in corpus]
