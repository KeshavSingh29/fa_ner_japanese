import json 
import logging
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    set_seed,
    EarlyStoppingCallback
)
from services.utils import (
    dir_path, 
    file_downloader, 
    create_label_id, 
    normalize_data, 
    process_token_labels, 
    adjust_labels, 
    create_train_val_test_data, 
    process_data2tags, 
    evaluate_result, 
    save_result, 
    save_dataset
)


logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

DIR_NAME = "ner_data"
FILE_NAME = "ner"
# using the latest bert base model 
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")

def main():

    # Set seed
    set_seed(1234)

    # Download data
    file_downloader(dir_name=DIR_NAME, file_name=FILE_NAME)

    # Read data
    with open(f"{dir_path}/{DIR_NAME}/{FILE_NAME}.json", "r") as f:
        ner_data = json.load(f)
    logger.info(f"Total Data points: {len(ner_data)}")

    # Normalize data
    n_data = normalize_data(data=ner_data)

    # Create Labels 
    label_id, id_label = create_label_id(data=n_data)

    # Create features
    feature_data = process_token_labels(json_data=n_data, label_id=label_id, tokenizer=tokenizer)

    # Adjust labels with SEP/CLS
    adjusted_feature_data = adjust_labels(data=feature_data)

    # Data Split
    train_val_test_split = create_train_val_test_data(data=adjusted_feature_data)
    
    # Save dataset
    save_dataset(data_features=train_val_test_split)

    # Define model
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding='longest')
    model = AutoModelForTokenClassification.from_pretrained(
        "tohoku-nlp/bert-base-japanese-v3", num_labels=len(label_id), id2label=id_label, label2id=label_id
    )

    # Training arg + Trainer
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        do_predict=True,
        output_dir="model_checkpoint",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_steps=250,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_val_test_split["train"],
        eval_dataset=train_val_test_split["val"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    logger.info("--------START MODEL TRAINING---------")
    trainer.train()
    trainer.save_model("ner_model")

    # Load best model
    inference_model = AutoModelForTokenClassification.from_pretrained("ner_model")
    
    # Make predictions 
    logger.info("--------START MODEL INFERENCE---------")
    predictions, actual = process_data2tags(test_data=train_val_test_split["test"], train_model=model, inference_model=inference_model, tokenizer=tokenizer)

    # Result
    result = evaluate_result(pred_labels=predictions, true_labels=actual)
    logger.info(result)

    # Save Result
    save_result(result_dict=result)



if __name__ == "__main__":
    main()