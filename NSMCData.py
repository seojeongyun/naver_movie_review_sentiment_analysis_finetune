import pandas as pd
import urllib
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm

class NSMCDataset(Dataset):
    def __init__(self, task):
        self.task = task
        self.data = self.get_download_data()

        # 일부 값중에 NaN이 있음...
        self.data = pd.read_csv(self.data, sep='\t').dropna(axis=0)

        # 중복제거
        self.data.drop_duplicates(subset=['document'], inplace=True)

        # Set Tokenizer from hugging face
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

    def get_download_data(self):
        if self.task == 'train':
            train_data = urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                                   filename="./data/ratings_train.txt")
            return train_data

        else :
            test_data = urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                                   filename="./data/ratings_test.txt")
            return test_data
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:3].values
        text = row[0]
        y = row[1]

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            pad_to_max_length='max_length',
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y