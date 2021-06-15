import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--one_hot_feature_len', type=int, default=52)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--eval_batch_size', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=768)
parser.add_argument('--gradient_accumulation_steps', default=6, type=int)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--warmup_steps', default=320, type=int)
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--bert_model_path', type=str, default='/home/zhchen/bert-base-uncased')
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--eval', type=bool, default=False)
parser.add_argument('--predict', type=bool, default=True)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--save_path', type=str, default='model/clothing_classification.pt')

args = parser.parse_args()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class ReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(
        reviews=df['review_text'].to_numpy(),
        targets=df['fit_label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


class TestReviewDataset(Dataset):

    def __init__(self, reviews, tokenizer, max_len):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


class ClothingClassifier(nn.Module):

    def __init__(self, n_classes):
        super(ClothingClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_model_path)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


def convert_sentence_to_features_bert(sentence, tokenizer=BertTokenizer):
    encoded_dict = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=args.max_seq_len,
        padding='max_length',
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoded_dict['input_ids'].flatten(), encoded_dict['attention_mask'].flatten()


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu())
            real_values.extend(targets.cpu())
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    print("Classification Report:\n ", classification_report(np.array(real_values), np.array(predictions)))
    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()

    predictions = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)

    predictions = torch.stack(predictions).cpu()
    return predictions


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if args.train or args.eval:
        df = pd.read_csv('data/train.txt', sep=',')
        df['fit_label'] = df['fit'].apply(
            lambda x: 0 if x == "small" else (2 if x == 'large' else (1 if x == 'fit' else x)))
        train_dataset, eval_dataset = np.split(
            df.sample(frac=1), [int(0.8 * len(df))])
        train_data_loader = create_data_loader(
            train_dataset, tokenizer, args.max_seq_len, args.batch_size)
        eval_data_loader = create_data_loader(
            eval_dataset, tokenizer, args.max_seq_len, args.batch_size)
        loss_fn = nn.CrossEntropyLoss().to(device)

    if args.train:
        model = ClothingClassifier(args.num_classes)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
        total_steps = len(train_data_loader) * args.epoch

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        history = defaultdict(list)
        best_accuracy = 0
        for epoch in range(args.epoch):

            print(f'Epoch {epoch + 1}/{args.epoch}')
            print('-' * 10)

            train_acc, train_loss = train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(train_dataset)
            )

            print(f'Train loss {train_loss} accuracy {train_acc}')
            val_acc, val_loss = eval_model(
                model,
                eval_data_loader,
                loss_fn,
                device,
                len(eval_dataset)
            )

            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model, args.save_path)
                best_accuracy = val_acc

    if args.eval:
        model = torch.load(args.save_path)
        val_acc, val_loss = eval_model(
            model,
            eval_data_loader,
            loss_fn,
            device,
            len(eval_dataset)
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
    if args.predict:
        logger.info(args.__dict__)
        model = torch.load(args.save_path)
        test_df = pd.read_csv('data/test.txt')
        test_dataset = TestReviewDataset(test_df['review_text'], tokenizer, args.max_seq_len)
        test_dataloder = DataLoader(test_dataset, args.batch_size, num_workers=4)
        preds = get_predictions(model, test_dataloder)
        preds = preds.tolist()
        with open('data/preds.txt', 'w', ) as f:
            for pred in preds:
                if pred == 0:
                    f.write('small\n')
                elif pred == 1:
                    f.write('fit\n')
                else:
                    f.write('large\n')
