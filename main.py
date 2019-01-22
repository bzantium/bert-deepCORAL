"""Main script for ADDA."""

from params import param
from core import train_src, eval_tgt
from models import BERTEncoder, BERTClassifier
from utils import read_data, get_data_loader, init_model, init_random_seed
from pytorch_pretrained_bert import BertTokenizer
import argparse
import numpy as np
import torch

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="dvd", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify tgt dataset")

    parser.add_argument('--seqlen', type=int, default=50,
                        help="Specify maximum sequence length")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify batch size")

    parser.add_argument('--num_epochs', type=int, default=5,
                        help="Specify the number of epochs for training")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for training")

    parser.add_argument('--eval_step', type=int, default=1,
                        help="Specify eval step size for training")

    parser.add_argument('--save_step', type=int, default=100,
                        help="Specify save step size for training")

    args = parser.parse_args()

    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seqlen: " + str(args.seqlen))
    print("batch_size: " + str(args.batch_size))
    print("num_epochs: " + str(args.num_epochs))
    print("log_step: " + str(args.log_step))
    print("eval_step: " + str(args.eval_step))
    print("save_step: " + str(args.save_step))

    # init random seed
    init_random_seed(param.manual_seed)

    # preprocess data
    print("=== Processing datasets ===")
    src_train = read_data('./data/processed/' + args.src + '/train.txt')
    src_test = read_data('./data/processed/' + args.src + '/test.txt')
    tgt_train = read_data('./data/processed/' + args.tgt + '/train.txt')
    tgt_test = read_data('./data/processed/' + args.tgt + '/test.txt')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    src_train_sequences = []
    src_test_sequences = []
    tgt_train_sequences = []
    tgt_test_sequences = []
    tgt_sequences = []

    for i in range(len(src_train.review)):
        tokenized_text = tokenizer.tokenize(src_train.review[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        src_train_sequences.append(indexed_tokens)

    for i in range(len(src_test.review)):
        tokenized_text = tokenizer.tokenize(src_test.review[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        src_test_sequences.append(indexed_tokens)

    for i in range(len(tgt_train.review)):
        tokenized_text = tokenizer.tokenize(tgt_train.review[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tgt_train_sequences.append(indexed_tokens)
        tgt_sequences.append(indexed_tokens)

    for i in range(len(tgt_test.review)):
        tokenized_text = tokenizer.tokenize(tgt_test.review[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tgt_test_sequences.append(indexed_tokens)
        tgt_sequences.append(indexed_tokens)

    # load dataset
    src_data_loader = get_data_loader(src_train_sequences, src_train.label, args.batch_size, args.seqlen)
    src_data_loader_eval = get_data_loader(src_test_sequences, src_test.label, args.batch_size, args.seqlen)
    tgt_data_loader = get_data_loader(tgt_train_sequences, tgt_train.label, args.batch_size, args.seqlen)
    tgt_data_loader_eval = get_data_loader(tgt_test_sequences, tgt_test.label, args.batch_size, args.seqlen)
    tgt_data_loader_all = get_data_loader(tgt_sequences, np.concatenate((tgt_train.label, tgt_test.label)),
                                          args.batch_size, args.seqlen)

    # load models
    encoder = BERTEncoder()
    classifier = BERTClassifier()

    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        classifier = torch.nn.DataParallel(classifier)

    encoder = init_model(encoder,
                             restore=param.encoder_restore)
    classifier = init_model(classifier,
                                restore=param.classifier_restore)
    # freeze source encoder params
    if torch.cuda.device_count() > 1:
        for params in encoder.module.encoder.embeddings.parameters():
            params.requires_grad = False
    else:
        for params in encoder.encoder.embeddings.parameters():
            params.requires_grad = False

    # train source model
    print("=== Training classifier for source domain ===")
    encoder, classifier = train_src(
        args, encoder, classifier, src_data_loader, tgt_data_loader, src_data_loader_eval)

    # eval target encoder on lambda0.1 set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> deepCORAL evaluation <<<")
    eval_tgt(encoder, classifier, tgt_data_loader_all)
