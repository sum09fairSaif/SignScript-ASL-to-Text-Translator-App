# Fine-tunes Flan-T5-small on ASLG-PC12 gloss->English pairs.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from translator_config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_EN_LEN, MAX_GLOSS_LEN, MODEL_NAME,
    PREPARED_DATA_DIR, TASK_PREFIX, TRANSLATOR_MODELS_DIR,
)


class GlossTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        inputs = (TASK_PREFIX + df['gloss']).tolist()
        targets = df['text'].tolist()
        model_inputs = tokenizer(inputs, max_length=MAX_GLOSS_LEN, truncation=True)
        labels = tokenizer(text_target=targets, max_length=MAX_EN_LEN, truncation=True)
        self.input_ids = model_inputs['input_ids']
        self.attention_mask = model_inputs['attention_mask']
        self.labels = labels['input_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }


def load_split(name):
    path = os.path.join(PREPARED_DATA_DIR, f'{name}.parquet')
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {path} -- run prepare_data.py first")
    return pd.read_parquet(path)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_df = load_split('train')
    val_df = load_split('val')

    train_dataset = GlossTextDataset(train_df, tokenizer)
    val_dataset = GlossTextDataset(val_df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    os.makedirs(TRANSLATOR_MODELS_DIR, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=TRANSLATOR_MODELS_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        logging_steps=100,
        report_to='none',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    final_dir = os.path.join(TRANSLATOR_MODELS_DIR, 'final')
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved fine-tuned model to {final_dir}")

    # Qualitative sanity check: a few sample translations from the val set.
    # Fluency here is bounded by ASLG-PC12's synthetic (rule-generated, not
    # real-signer) gloss style -- this is a first-pass baseline, not a
    # fluency bar to hit.
    print("\nSample translations:")
    model.eval()
    for i in range(min(5, len(val_df))):
        gloss = val_df.iloc[i]['gloss']
        reference = val_df.iloc[i]['text']
        inputs = tokenizer(TASK_PREFIX + gloss, return_tensors='pt', max_length=MAX_GLOSS_LEN, truncation=True)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=MAX_EN_LEN)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"  gloss:     {gloss}")
        print(f"  reference: {reference}")
        print(f"  predicted: {prediction}\n")


if __name__ == '__main__':
    main()
