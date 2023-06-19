import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from copy import deepcopy


class CNN_LSTM_NLP(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes=10000,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 lstm_unit_shape=128,
                 output_shape=128,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            vocab_size (int)
            num_classes(int): num classes in the end (usually bigger than what we have) Default:10000
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            output_shape (int): Number of classes. Default: 128
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_LSTM_NLP, self).__init__()

        self.embed_dim = embed_dim
        # max_norm (float, optional) – If given,
        # each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=self.embed_dim)
        ################ Conv Network##############################
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=0.05)
        self.relu = nn.ReLU()

        ################ LSTM Network ##############################

        LSTM_UNITS = lstm_unit_shape
        DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
        self.lstm1 = nn.LSTM(self.embed_dim, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.fc1 = nn.Linear(sum(num_filters) + DENSE_HIDDEN_UNITS, 2048)
        self.fc2 = nn.Linear(2048, output_shape)

    def forward(self, input_ids):
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        input_ids = input_ids.long()

        x_embed = self.embedding(input_ids)  # .float()

        # CNN PART:
        #####################
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        # Compute logits. Output shape: (b, n_classes)
        result_CNN = self.dropout(x_fc)

        # LSTM PART:
        #####################

        h_lstm1, _ = self.lstm1(x_embed)
        h_lstm2, _ = self.lstm2(h_lstm1)
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        result_lstm = h_conc + h_conc_linear1 + h_conc_linear2

        res = torch.cat((result_CNN, result_lstm), 1)

        res = self.fc1(res)
        res_dropped = self.dropout2(res)
        finalo = self.fc2(res_dropped)
        return finalo


from torch.nn.utils import clip_grad_norm_
import torch
import tqdm


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

from torch.optim.lr_scheduler import StepLR

def TM(model, optimizer, criterion, train_dataloader, epochs, device):
    model.train()
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(epochs):
        scheduler.step()
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())

        running_loss = []
        for step, (anc, pos, neg) in enumerate(tqdm.tqdm(train_dataloader, desc="Training", leave=False)):
            anc = anc.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            optimizer.zero_grad()
            anchor_out = model(anc)
            positive_out = model(pos)
            negative_out = model(neg)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))

    return model


"""
def train_model(model,optimizer,criterion, train_dataloader, epochs, device):

   
    '''
    def l_infinity(x1, x2):
        return torch.max(torch.abs(x1 - x2), dim=1).values

    # triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=l_infinity, margin=1500)
    '''
    # Tracking best validation accuracy
    val_losses = []
    train_losses = []

    # Start training loop
    print("Start training...\n")

    for epoch_i in range(epochs):
        print('epoch_i: ', epoch_i)

        anc_m.train()
        pos_m.train()
        neg_m.train()

        for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
            # Load batch to GPU
            anc, pos, neg = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients

            # Perform a forward pass. This will return output.
            output_anc = model(anc)
            output_pos = model(pos)
            output_neg = model(neg)

            # Compute loss and accumulate the loss values

            '''
            def l_infinity(x1, x2):
                return torch.max(torch.abs(x1 - x2), dim=1).values
            print('anc: ', output_anc)
            print('anc: ', output_anc.shape)

            print(f' anc pos: {l_infinity(output_anc, output_pos)}')
            print(f' anc neg: {l_infinity(output_anc, output_neg)}')
            print(f' neg pos: {l_infinity(output_neg, output_pos)}')
            '''
            loss = triplet_loss(anchor=output_anc, positive=output_pos, negative=output_neg)
            tr_loss = loss.item()
            train_losses.append(tr_loss)

            # Perform a backward pass to calculate gradients
            loss.backward()
            # clip_grad_norm_(model.parameters(), 1.)

            # Update parameters
            optimizer_anc_m.step()
            optimizer_pos_m.step()
            optimizer_neg_m.step()

            anc_m.zero_grad()
            pos_m.zero_grad()
            neg_m.zero_grad()

        print('train_losses[-1]:', train_losses[-1])
        print('avg loss: ', sum(train_losses) / len(train_losses))

    return anc_m
"""

'''
def train_model(model, optimizer, train_dataloader, val_dataloader, test_generator, y_train_tokenize, num_noise, epochs,
                device):
    model_name = model.__class__.__name__
    # Tracking best validation accuracy
    val_losses = []
    train_losses = []

    # Start training loop
    print("Start training...\n")

    train_mean_epoch = {}
    valid_mean_epoch = {}
    train_last_epoch = {}
    valid_last_epoch = {}

    for epoch_i in range(epochs):

        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return output.
            output = model(b_input_ids)

            # Compute loss and accumulate the loss values

            loss = model.direct_loss(output, b_labels.unsqueeze(-1), num_noise)
            tr_loss = loss.item()
            train_losses.append(tr_loss)

            # Perform a backward pass to calculate gradients
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.)

            # Update parameters
            optimizer.step()

        last_tr_loss = train_losses[-1]
        mean_tr_loss = sum(train_losses) / len(train_losses)

        # Calculate the average loss over the entire training data

        # =======================================
        #               VALIDATION
        # =======================================
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                b_input_ids, b_labels = tuple(t.to(device) for t in batch)
                output = model(b_input_ids)
                val_loss = model.direct_loss(output, b_labels.unsqueeze(-1), num_noise)
                val_losses.append(val_loss)

            last_val_loss = val_losses[-1]
            mean_val_loss = sum(val_losses) / len(val_losses)

        if epoch_i % 10 == 0:
            f = open("results/{}.txt".format(model_name), "a")
            f.write('step {}, last train loss : {}, mean train loss: {}, last val loss: {}, mean val loss: {} '.format(
                epoch_i, last_tr_loss, mean_tr_loss, last_val_loss, mean_val_loss))
            f.write("\n")
            f.close()

        if epoch_i % 50 == 0 and epoch_i > 1:
            t1, t5, t10, t20, num = find_most_sim(model, y_train_tokenize, test_generator)
            f = open("results/{}.txt".format(model_name), "a")
            f.write('t1 : {}, t5: {}, t10: {}, t20: {}'.format(t1, t5, t10, t20))
            f.write("\n")
            f.close()

        print(
            'step {}, last train loss : {:.4f}, mean train loss: {:.4f}, last val loss: {:.4f}, mean val loss: {:.4f} '.format(
                epoch_i, last_tr_loss, mean_tr_loss, last_val_loss, mean_val_loss))

        train_mean_epoch[epoch_i] = mean_tr_loss
        valid_mean_epoch[epoch_i] = mean_val_loss
        train_last_epoch[epoch_i] = last_tr_loss
        valid_last_epoch[epoch_i] = last_val_loss

    t1, t5, t10, t20, num = find_most_sim(model, y_train_tokenize, test_generator)
    f = open("results/{}.txt".format(model_name), "a")
    f.write('t1 : {}, t5: {}, t10: {}, t20: {}'.format(t1, t5, t10, t20))
    f.write("\n")
    f.write('############## END OF TRAINING ##############')
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.close()
    return train_mean_epoch, valid_mean_epoch, train_last_epoch, valid_last_epoch

'''
