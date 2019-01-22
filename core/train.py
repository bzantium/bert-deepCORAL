"""Pre-train encoder and classifier for source dataset."""

import torch
import torch.nn as nn
from params import param
from utils import save_model
import torch.optim as optim


def CORAL(source, target):
    d = source.data.size(1)

    # source covariance
    xms = source - torch.mean(source, 0, keepdim=True)
    xcs = xms.t() @ xms

    # target covariance
    xmt = target - torch.mean(target, 0, keepdim=True)
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.sum(torch.mul((xcs - xct), (xcs - xct)))
    loss = loss/(4*d*d)

    return loss


def train_src(args, encoder, classifier, src_data_loader, tgt_data_loader, data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate,
                           betas=(param.beta1, param.beta2))
    criterion = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    ####################
    # 2. train network #
    ####################

    for epoch in range(args.num_epochs):
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((src_reviews, src_labels), (tgt_reviews, _)) in data_zip:

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            src_mask = (src_reviews != 0).long()
            tgt_mask = (tgt_reviews != 0).long()
            src_feat = encoder(src_reviews, src_mask)
            tgt_feat = encoder(tgt_reviews, tgt_mask)
            src_preds = classifier(src_feat)
            class_loss = criterion(src_preds, src_labels)
            coral_loss = CORAL(src_feat, tgt_feat)
            loss = class_loss + coral_loss

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.3d/%.3d] Step [%.2d/%.2d]: class_loss=%.4f coral_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len(src_data_loader),
                         class_loss.item(),
                         coral_loss.item()))

        # eval model on lambda0.1 set
        if (epoch + 1) % args.eval_step == 0:
            # print('Epoch [{}/{}]'.format(epoch + 1, param.num_epochs_pre))
            eval_src(encoder, classifier, src_data_loader)
            eval_src(encoder, classifier, data_loader_eval)
            print()

        # save model parameters
        if (epoch + 1) % args.save_step == 0:
            save_model(encoder, "CORAL-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, "CORAL-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "CORAL-encoder-final.pt")
    save_model(classifier, "CORAL-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, labels) in data_loader:
        mask = (reviews > 0).long()
        preds = classifier(encoder(reviews, mask))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
