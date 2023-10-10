import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
from NSMCData import NSMCDataset
from transformers import AutoModel

class Trainer:
    def __init__(self, task):
        self.task = task
        self.device = "cuda:1"
        #
        self.epochs = 5
        self.batch_size = 8
        #
        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()
        #
        self.dataloader = self.get_loader()
        #
        self.optimizer = self.get_optim()
        #
        self.nums = 0
        self.total_loss = 0.0
        self.correct = 0
        self.total = 0
        self.batches = 0
        #
        self.losses = []
        self.accuracy = []


    def get_loader(self):
        NSMC = NSMCDataset(self.task, self.tokenizer)
        data_loader = torch.utils.data.DataLoader(NSMC,
                                                  shuffle=True,
                                                  batch_size=self.batch_size)
        return data_loader

    def get_model(self):
        model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(self.device)
        return model

    def get_optim(self):
        return torch.optim.AdamW(self.model.parameters(), lr=5e-6)

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")
        return tokenizer

    def train_one_epoch(self):
        for input_ids_batch, attention_masks_batch, y_batch in self.dataloader:
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
        self.losses = []
        self.accuracy = []

        for i in range(self.epochs):
            self.model.train()
            self.train_one_epoch()
            #
            torch.save(self.model,'./ckpt/last_ckpt.pt')
            #
            self.losses.append(self.total_loss)
            self.accuracy.append(self.correct.float() / self.total)
            print("Train Loss:", self.total_loss, "Accuracy:", self.correct.float() / self.total)
            #
            self.total_loss = 0.0
            self.correct = 0
            self.total = 0
            self.batches = 0

            self.upload_model_to_hugging_face()

    def eval(self):
        self.model.eval()

        test_correct = 0
        test_total = 0

        for input_ids_batch, attention_masks_batch, y_batch in self.dataloader:
            y_batch = y_batch.to(self.device)
            y_pred = self.model(input_ids_batch.to(self.device), attention_mask=attention_masks_batch.to(self.device))[0]
            _, predicted = torch.max(y_pred, 1)
            test_correct += (predicted == y_batch).sum()
            test_total += len(y_batch)

        print("Accuracy:", test_correct.float() / test_total)


    def print_loss(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.xlabel('Epoch')
        plt.ylabel('{}_Loss'.format(self.task))
        plt.plot(self.nums, self.losses)
        plt.subplot(1, 2, 2)
        plt.xlabel('Epoch')
        plt.ylabel('{}}_acc'.format(self.task))
        plt.plot(self.nums, self.accuracy)
        plt.show()

    def upload_model_to_hugging_face(self):
        # Huggingface Access Token
        ACCESS_TOKEN = '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'

        # Upload to Huggingface
        self.model.push_to_hub('NSMC_finetune_jy', use_temp_dir=True, use_auth_token=ACCESS_TOKEN)
        self.tokenizer.push_to_hub('NSMC_finetune_jy', use_temp_dir=True, use_auth_token=ACCESS_TOKEN)