import os 
import json
import torch
import noyaki
import logging
import warnings
import requests
import evaluate
import unicodedata
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))

def file_downloader(dir_name: str, file_name: str = "ner", url: str = "https://github.com/stockmarkteam/ner-wikipedia-dataset/raw/main/ner.json") -> None:
    """
    Downloads a file from a given absolute file url
    """
    response = requests.get(url)
    if response.status_code == 200:
        if not os.path.isdir(f"{dir_path}/{dir_name}"):
            os.mkdir(f"{dir_path}/{dir_name}")
            with open(f"{dir_path}/{dir_name}/{file_name}.json", "wb") as f:
                f.write(response.content)
        else:
            with open(f"{dir_path}/{dir_name}/{dir_name}.json", "wb") as f:
                f.write(response.content)
        logger.info("File downloaded successfully.")
    else:
        logger.info("Failed to download file. Status code:", response.status_code)


def normalize_data(data: list[dict]) -> list[dict]:
    """
    Normalize half width and full width characters
    """
    for d in data:
        d['text'] = unicodedata.normalize('NFKC', d['text'])
    return data


def create_label_id(data: list[dict]) -> dict:
    """
    Create dict of labels id for NER task
    """
    unique_labels = set("O") # lebel 'O' for untagged tokens
    for json_data in data:
        if json_data['entities']:
            unique_labels.add(json_data['entities'][0]['type'])
    label_id = {k:v for v,k in enumerate(unique_labels)}
    id_label = {v: k for v,k in enumerate(unique_labels)}
    logger.info(f"Total NER Tags: {len(unique_labels)}\n{label_id}")

    return label_id, id_label


def process_token_labels(json_data: list[dict], label_id: dict, tokenizer) -> list:
    """
    Retuns input features for NER model. 
    Can also be used to create dataset
    """
    labeled_data = []
    for _, unit in tqdm(enumerate(json_data)):
        """ 
        tokenizer will tokenize differently (using WordPiece subword tokenization for NER task by default)
        eg: 
        text: "SPRiNGSと最も仲の良いライバルグループ。" 
        tokenized_text: ['SP', '##R', '##i', '##N', '##GS', 'と', '最も', '仲', 'の', '良い', 'ライバル', 'グループ', '。']
        """
        tokenized_text = tokenizer.tokenize(unit['text'])
        features = tokenizer(unit['text']) 
        spans = []
        for entity in unit["entities"]:
            span_list = []
            span_list.extend(entity["span"])
            span_list.append(entity["type"])
            spans.append(span_list)
        label = noyaki.convert(tokenized_text, spans, subword="##")
        # noyaki uses BILOU tags by default, we can remove them
        labels = [label_id[x.split("-")[-1]] for x in label]
        labeled_data.append({"text": unit['text'],"tokens": tokenized_text, "labels": labels, "input_ids": features['input_ids'], "token_type_ids": features['token_type_ids'], 'attention_mask': features['attention_mask']})
    logger.info("Finished creating features for NER data")
    return labeled_data


def adjust_labels(data: list[dict]) -> list[dict]:
    # Bert adds cls and sep tokens , we can assign a -100 label to them
    for item in data:
        labels = [-100] + item['labels'] + [-100]
        item['labels'] = labels
    return data


def create_train_val_test_data(data: list[dict], train_size : int = 0.8):
    """
    Create train val and test splits
    """
    full_dataset = Dataset.from_list(data)
    train_valtest_split = full_dataset.train_test_split(test_size=1-train_size, shuffle=True)
    val_test_split = train_valtest_split['test'].train_test_split(test_size=0.5, shuffle=True)
    train_val_test_split = DatasetDict({'train': train_valtest_split['train'], 'val': val_test_split['train'], 'test': val_test_split['test']})
    logger.info(f"Create Train-Val-Test Split\n {train_val_test_split}")
    return train_val_test_split


def save_dataset(data_features) -> None:
    """
    Save dataset to disk
    """
    data_features.save_to_disk(f"{dir_path}/ner_partition_data")


def inference(data_piece, inference_model, tokenizer) -> list:
    """
    Returns raw ids of predicted class, you may need to map them to correct label using id_label dict
    """
    inputs = tokenizer(data_piece['text'], return_tensors="pt")
    with torch.no_grad():
        logits = inference_model(**inputs).logits[0]
    pred = np.argmax(logits.detach().numpy(), axis=-1)
    predicted_token_class = [inference_model.config.id2label[t.item()] for t in pred]
    return predicted_token_class[1:-1]

def process_data2tags(test_data, train_model, inference_model, tokenizer):
    """
    Returns correct and predicted NER tags
    """
    # get correct labels
    y_true, y_pred = [], []
    for unit in tqdm(test_data):
        labels = unit['labels'][1:-1]
        token_class = [train_model.config.id2label[t] for t in labels]
        y_true.append(token_class)
    # get predicted labels
        y_pred.append(inference(data_piece=unit, inference_model=inference_model, tokenizer=tokenizer))
    return y_pred, y_true

def token_aggregator(tokens: list[str],labels: list[str]) -> list[dict]:
    """
    Returns text and corresponding tags with correct span
    Example:
    Input: 
        tokens=['SP','##R','##i','##N','##GS','と',..]
        tags=["その他の組織名","その他の組織名","その他の組織名","その他の組織名","その他の組織名","O",...])

    Output:
        [{'tag': 'その他の組織名', 'text': 'SPRiNGS'},
        {'tag': 'O', 'text': 'と最も仲の良いライバルグループ。'}]
    """

    entities = []
    slow, fast  = 0, 0
    end = len(labels)
    while slow < end:
        if fast != end -1:
            if labels[slow] == labels[fast]:
                fast += 1
            else:
                entities.append({"tag":labels[slow], "text": "".join(tokens[slow:fast]).replace("#","")})
                slow = fast
                fast += 1
        else:
            entities.append({"tag":labels[slow], "text": "".join(tokens[slow:]).replace("#","")})
            return entities
    return entities


def evaluate_result(pred_labels, true_labels):
    seqeval = evaluate.load('seqeval')
    result = seqeval.compute(references = true_labels, predictions=pred_labels)
    return result

def convert_int64_to_int(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    return obj

def save_result(result_dict: dict):
    with open("result.txt", "w") as file:
        json.dump(result_dict, file, indent=4, ensure_ascii=False, default=convert_int64_to_int)

if __name__ == "__main__":
    pass