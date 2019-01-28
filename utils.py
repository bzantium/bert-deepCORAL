import os
import random
import numpy as np
import pandas as pd
import torch
from lxml import etree
import xml.etree.ElementTree as ET
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from pytorch_pretrained_bert import BertTokenizer
from params import param
import re

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def XML2Array(neg_path, pos_path):
    parser = etree.XMLParser(recover=True)
    reviews = []
    negCount = 0
    posCount = 0
    labels = []
    regex = re.compile(r'[\n\r\t+]')

    neg_tree = ET.parse(neg_path, parser=parser)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review_text'):
        text = regex.sub(" ", rev.text)
        reviews.append(text)
        negCount += 1
    labels.extend(np.zeros(negCount, dtype=int))

    pos_tree = ET.parse(pos_path, parser=parser)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review_text'):
        text = regex.sub(" ", rev.text)
        reviews.append(text)
        posCount += 1
    labels.extend(np.ones(posCount, dtype=int))

    return reviews, labels


def TSV2Array(path):
    data = pd.read_csv(path, delimiter='\t')
    reviews, labels = data.reviews.values.tolist(), data.labels.values.tolist()
    return reviews, labels


def review2seq(reviews):
    sequences = []
    for i in range(len(reviews)):
        tokens = tokenizer.tokenize(reviews[i])
        sequence = tokenizer.convert_tokens_to_ids(tokens)
        sequences.append(sequence)
    return sequences


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore=None):

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(param.model_root):
        os.makedirs(param.model_root)
    torch.save(net.state_dict(),
               os.path.join(param.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(param.model_root, filename)))


def get_data_loader(sequences, labels, batch_size, maxlen=None):
    # dataset and data loader
    text_dataset = TextDataset(sequences, labels, maxlen)

    text_data_loader = DataLoader(
        dataset=text_dataset,
        batch_size=batch_size,
        shuffle=True)

    return text_data_loader


class TextDataset(Dataset):
    def __init__(self, sequences, labels, maxlen):

        seqlen = max([len(sequence) for sequence in sequences])

        if maxlen is None or maxlen > seqlen:
            maxlen = seqlen

        seq_data = list()
        for sequence in sequences:
            sequence.insert(0, 101) # insert [CLS] token
            sequence.append(102) # insert [SEP] token
            seqlen = len(sequence)
            if seqlen < maxlen:
                sequence.extend([0] * (maxlen-seqlen))
            else:
                sequence = sequence[:maxlen]
            seq_data.append(sequence)

        self.data = torch.LongTensor(seq_data).cuda()
        self.labels = torch.LongTensor(labels).cuda()
        self.dataset_size = len(self.data)

    def __getitem__(self, index):
        review, label = self.data[index], self.labels[index]
        return review, label

    def __len__(self):
        return self.dataset_size
