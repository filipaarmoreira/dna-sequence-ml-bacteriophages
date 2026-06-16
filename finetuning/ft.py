
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    logging,
)

CHECKPOINT ="LongSafari/hyenadna-small-32k-seqlen-hf"
MAX_LENGTH  = 32_000
DATASET_PATH = "dataset/processed/final_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2



df = pd.read_csv(DATASET_PATH)


#replace ambiguous characters (M, Y, R, K, W, S) with N
def clean_sequence(seq):
    return re.sub(r'[^ACGTacgt]', 'N', seq)

sequences  = [clean_sequence(s) for s in df["Sequence"].tolist()]
labels_raw = df["Label"].tolist()


le     = LabelEncoder()
labels = le.fit_transform(labels_raw).tolist()
print(f"Classes: {list(le.classes_)} → {list(range(len(le.classes_)))}")

#80/20 stratified split
seq_train, seq_test, y_train, y_test = train_test_split(
    sequences, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)



tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    torch_dtype=torch.bfloat16,   
    device_map="auto",             
    trust_remote_code=True,
    num_labels=len(le.classes_),   
)


tokenized_train = tokenizer(seq_train, max_length=MAX_LENGTH, truncation=True, padding="max_length")["input_ids"]
tokenized_test  = tokenizer(seq_test,  max_length=MAX_LENGTH, truncation=True, padding="max_length")["input_ids"]


train_dataset = Dataset.from_dict({"input_ids": tokenized_train, "labels": y_train})
test_dataset  = Dataset.from_dict({"input_ids": tokenized_test,  "labels": y_test})

train_dataset.set_format("pt")   
test_dataset.set_format("pt")

# Training Arguments 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1_macro": f1_score(labels, preds, average="macro")}

args = {
    "output_dir":                   "results/phase3_e2e",
    "num_train_epochs":             5,
    "per_device_train_batch_size":  1,
    "gradient_accumulation_steps":  4,
    "gradient_checkpointing":       True,
    "learning_rate":                2e-5,
    "eval_strategy":                "epoch",
    "save_strategy":                "no",        # não guarda checkpoints
    "load_best_model_at_end":       False,       
    "seed":                         RANDOM_STATE,
}
training_args = TrainingArguments(**args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,         
    compute_metrics=compute_metrics,    
)

# Train 
result = trainer.train()



predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

f1 = f1_score(y_test, y_pred, average="macro")



# Classification report como dict
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("results/phase3_e2e_classification_report.csv")

# Matriz de confusão como CSV
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
cm_df.to_csv("results/phase3_e2e_confusion_matrix.csv")

# Resultados gerais
pd.DataFrame([{
    "model":      CHECKPOINT,
    "max_length": MAX_LENGTH,
    "epochs":     5,
    "f1_macro":   f1,
    "n_train":    len(seq_train),
    "n_test":     len(seq_test),
}]).to_csv("results/phase3_e2e_results.csv", index=False)


