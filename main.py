"""Main script for ADDA."""

from params import param
from core import train_src, eval_tgt
from models import BERTEncoder, BERTClassifier
from utils import XML2Array, blog2Array, review2seq, \
    get_data_loader, init_model
from sklearn.model_selection import train_test_split
import os
import argparse
from pytorch_pretrained_bert import BertTokenizer
import torch

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="dvd", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify tgt dataset")

    parser.add_argument('--random_state', type=int, default=42,
                        help="Specify random state")

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
    print("random_state: " + str(args.random_state))
    print("seqlen: " + str(args.seqlen))
    print("batch_size: " + str(args.batch_size))
    print("num_epochs: " + str(args.num_epochs))
    print("log_step: " + str(args.log_step))
    print("eval_step: " + str(args.eval_step))
    print("save_step: " + str(args.save_step))

    # preprocess data
    print("=== Processing datasets ===")
	reviews, labels = XML2Array(os.path.join('data', args.src, 'negative.parsed'),
                                os.path.join('data', args.src, 'positive.parsed'))

    src_X_train, src_X_test, src_Y_train, src_Y_test = train_test_split(reviews, labels,
                                                                        test_size=0.2,
                                                                        random_state=args.random_state)
    del reviews, labels

    if args.tgt == 'blog':
        tgt_X, tgt_Y = blog2Array(os.path.join('data', args.tgt, 'blog.parsed'))

    else:
        tgt_X, tgt_Y = XML2Array(os.path.join('data', args.tgt, 'negative.parsed'),
                                 os.path.join('data', args.tgt, 'positive.parsed'))

    src_X_train = review2seq(src_X_train)
    src_X_test = review2seq(src_X_test)
    tgt_X = review2seq(tgt_X)

    # load dataset
    src_data_loader = get_data_loader(src_X_train, src_Y_train, args.batch_size, args.seqlen)
    src_data_loader_eval = get_data_loader(src_X_test, src_Y_test, args.batch_size, args.seqlen)
    tgt_data_loader = get_data_loader(tgt_X, tgt_Y, args.batch_size, args.seqlen)


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
    eval_tgt(encoder, classifier, tgt_data_loader)
