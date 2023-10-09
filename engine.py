import pandas as pd
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
from NSMCData import NSMCDataset
class Trainer:
    def __init__(self, task):
        self.task = task
        self.device = "cuda:1"
        #
        self.epochs = 5
        self.batch_size = 16
        #
        self.model = self.get_model()
        #
        self.dataloader = self.get_loader()
        #
        self.optimizer = self.get_optim()
        #
        self.total_loss = 0.0
        self.correct = 0
        self.total = 0
        self.batches = 0


    def get_loader(self):
        NSMC = NSMCDataset(self.task)
        data_loader = torch.utils.data.DataLoader(NSMC,
                                                  shuffle=True,
                                                  batch_size=self.batch_size)
        return data_loader

    def get_model(self):
        model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(self.device)
        return model

    def get_optim(self):
        return AdamW(self.model.parameters(), lr=5e-6)

    def train_one_epoch(self):
        for input_ids_batch, attention_masks_batch, y_batch in tqdm(self.dataloader):
            self.optimizer.zero_grad()
            y_batch = y_batch.to(self.device)
            y_pred = self.model(input_ids_batch.to(self.device), attention_mask=attention_masks_batch.to(self.device))[0]
            loss = F.cross_entropy(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()

            self.total_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            self.correct += (predicted == y_batch).sum()
            self.total += len(y_batch)

            self.batches += 1
            if self.batches % 100 == 0:
                print("Batch Loss:", self.total_loss, "Accuracy:", self.correct.float() / self.total)


    def train(self):
        losses = []
        accuracy = []

        for i in range(self.epochs):
            self.model.train()
            self.train_one_epoch()
            #
            losses.append(self.total_loss)
            accuracy.append(self.correct.float() / self.total)
            print("Train Loss:", self.total_loss, "Accuracy:", self.correct.float() / self.total)
            #
            self.total_loss = 0.0
            self.correct = 0
            self.total = 0
            self.batches = 0

    def test(self):
