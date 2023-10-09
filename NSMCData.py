import os
import pandas as pd
import urllib

from torch.utils.data import Dataset

class NSMCDataset(Dataset):
    def __init__(self, task, tokenizer):
        self.task = task
        self.tokenizer = tokenizer
        #
        if os.path.exists("./data/ratings_{}.txt".format(self.task)):
            pass
        else:
            self.data = self.get_download_data()

        # 일부 값중에 NaN이 있음...
        self.data = pd.read_csv("./data/ratings_{}.txt".format(self.task), sep='\t').dropna(axis=0)

        # 중복제거
        self.data.drop_duplicates(subset=['document'], inplace=True)

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
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, 1:3].values
        text = row[0]
        y = row[1]

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
        #   padding = True
            pad_to_max_length='max_length',
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y