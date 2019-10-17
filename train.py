"""Adversarial adaptation to train target encoder."""

import torch
from utils import make_cuda
import torch.nn as nn
import param
import torch.optim as optim
from utils import save_model


def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    # source covariance
    tmp_s = torch.sum(source, dim=0, keepdim=True)
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
    # target covariance
    tmp_t = torch.sum(target, dim=0, keepdim=True)
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)
    return loss


def MMD(source, target):
    delta = source - target
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def train(args, encoder, classifier,
          src_data_loader, src_data_loader_eval,
          tgt_data_loader, tgt_data_loader_all):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(classifier.parameters()),
                           lr=param.c_learning_rate)

    ####################
    # 2. train network #
    ####################

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((src_reviews, src_mask, src_labels), (tgt_reviews, tgt_mask, _)) in data_zip:
            src_reviews = make_cuda(src_reviews)
            src_mask = make_cuda(src_mask)
            src_labels = make_cuda(src_labels)
            tgt_reviews = make_cuda(tgt_reviews)
            tgt_mask = make_cuda(tgt_mask)

            # extract and concat features
            src_feat = encoder(src_reviews, src_mask)
            tgt_feat = encoder(tgt_reviews, tgt_mask)
            src_preds = classifier(src_feat)

            # prepare real and fake label
            optimizer.zero_grad()
            cls_loss = CELoss(src_preds, src_labels)
            if args.method == 'coral':
                adapt_loss = CORAL(src_feat, tgt_feat)
            else: # args.method == 'mmd'
                adapt_loss = MMD(src_feat, tgt_feat)
            loss = cls_loss + args.alpha * adapt_loss

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f coral_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len(src_data_loader),
                         cls_loss.item(),
                         adapt_loss.item()))

        evaluate(encoder, classifier, src_data_loader)
        evaluate(encoder, classifier, src_data_loader_eval)
        evaluate(encoder, classifier, tgt_data_loader_all)

    save_model(encoder, param.encoder_path)
    save_model(classifier, param.classifier_path)

    return encoder, classifier


def evaluate(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, mask, labels) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        labels = make_cuda(labels)

        with torch.no_grad():
            feat = encoder(reviews, mask)
            preds = classifier(feat)
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    return acc
