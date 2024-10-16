import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the CSV file
data_df = pd.read_csv('./cartographic_analysis/output_agreement_levels.csv')

# Check unique labels in the DataFrame
print("Unique labels in the dataset:", data_df['label'].unique())

# Ensure the labels are in the expected format (e.g., integers)
data_df['label'] = data_df['label'].astype(int)

# Count entries for each label
label_counts = data_df['label'].value_counts()
print("Counts of each label:", label_counts)

# Adjust balancing logic
min_count = min(label_counts)

# If min_count is less than 1962, use min_count instead
data_df_temp_0 = data_df[data_df['label'] == 0].sample(min_count, random_state=42, replace=True)
data_df_temp_1 = data_df[data_df['label'] == 1].sample(min_count, random_state=42, replace=True)

# After sampling and concatenating
train_df_balanced = pd.concat([data_df_temp_0, data_df_temp_1], axis=0).reset_index(drop=True)

# Split the data into training, validation, and test sets
train_df = train_df_balanced.sample(frac=0.8, random_state=42)
val_df = train_df_balanced.drop(train_df.index).reset_index(drop=True)
test_df = data_df.drop(train_df_balanced.index).reset_index(drop=True)

# Model and Dataset setup
BERT = 'bert-base-uncased'
ROBERTA = 'roberta-base'
MODEL_NAME = ROBERTA

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set training parameters
TRAIN_BATCH_SIZE = 32  # Consider increasing if you have enough memory
VALID_BATCH_SIZE = 32
MAX_LEN = 256
EPOCHS = 10
LEARNING_RATE = 5e-5
NUM_LABELS = 2

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.len = len(df)
        self.df = df.reset_index(drop=True)  # Ensure indices are reset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        if index >= self.len:
            raise IndexError(f"Index {index} is out of bounds for DataFrame with length {self.len}.")

        text = self.df.iloc[index]['text']  # Use iloc for positional indexing
        text_id = self.df.iloc[index]['text_id']
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            truncation=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = torch.zeros_like(torch.tensor(ids))  # Set to zeros, same shape as ids

        target = self.df.iloc[index]['label']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'masks': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': token_type_ids,  # Use zeros tensor
            'targets': torch.tensor(target, dtype=torch.long),
            'text_ids': text_id
        }

    def __len__(self):
        return self.len

class HateSpeechClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(HateSpeechClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.in_features = self.bert.config.hidden_size  # Ensure we use hidden_size for the classifier
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(self.in_features, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get the outputs from the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # The outputs are typically a tuple, so handle it accordingly
        last_hidden_state = outputs[0]  # The last hidden states

        # Use the [CLS] token representation for classification
        cls_representation = last_hidden_state[:, 0, :]  # Get the representation for the first token

        # Apply dropout
        output = self.dropout(cls_representation)

        # Pass through the classifier
        output = self.classifier(output)
        return output

# Setup for training
def get_train_setup(train_df, val_df):
    train_data = TextDataset(train_df, tokenizer, MAX_LEN)
    val_data = TextDataset(val_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=VALID_BATCH_SIZE, shuffle=False)

    model = HateSpeechClassifier(MODEL_NAME, NUM_LABELS)
    model.to(device)

    num_training_steps = EPOCHS * len(train_loader)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=100,
    )

    return train_loader, val_loader, model, loss_function, optimizer, scheduler

# Training function
def train(model, model_output_path, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=5, evaluation=False):
    print("Training...\n")
    train_values = []

    best_val_loss = None
    best_val_accuracy = None

    for epoch in range(epochs):
        print(f"Length of Train Dataloader: {len(train_dataloader)}")
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        train_loss = []
        val_loss = None
        val_accuracy = None

        model.train()
        epoch_train_values = torch.empty(0, dtype=torch.float)

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                model.zero_grad()
                input_ids = batch['ids'].to(device, dtype=torch.long)
                attention_mask = batch['masks'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                targets = batch['targets'].to(device, dtype=torch.long)
                text_ids = batch['text_ids'].unsqueeze(1)

                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(outputs, targets)
                train_loss.append(loss.item())

                batch_loss += loss.item()
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                # Cartography
                softmax_values = torch.nn.functional.softmax(outputs, dim=1)
                batch_train_values = torch.cat((text_ids, softmax_values.cpu()), dim=1)
                epoch_train_values = torch.cat((epoch_train_values, batch_train_values), dim=0)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': batch_loss / batch_counts})

                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    time_elapsed = time.time() - t0_batch
                    print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

        if evaluation:
            val_loss, val_accuracy = evaluate(model, loss_fn, val_dataloader)
            time_elapsed = time.time() - t0_epoch

            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\nBest model at epoch: {epoch + 1}")
                torch.save(model.state_dict(), model_output_path)

            if not best_val_accuracy or val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

            print(f"\n Validation Loss: {val_loss:^10.6f} | Validation Accuracy: {val_accuracy:^9.2f} | Time elapsed: {time_elapsed:^9.2f}\n")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"{epoch + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {best_val_loss:^10.6f} | {best_val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        print("-" * 70)

        train_values.append(epoch_train_values)
        print("\n")

    print("Training complete!")
    return train_values, best_val_loss, best_val_accuracy

# Evaluation function with tqdm
# Updated evaluation function with CPU conversion
def evaluate(model, loss_fn, val_dataloader):
    model.eval()
    val_accuracy = []
    val_loss = []

    with tqdm(total=len(val_dataloader), desc="Evaluating", unit="batch") as pbar:
        for batch in val_dataloader:
            input_ids = batch['ids'].to(device, dtype=torch.long)
            attention_mask = batch['masks'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(outputs, targets)

            val_loss.append(loss.item())
            
            # Move outputs to CPU before calculating accuracy
            val_accuracy.append((outputs.argmax(dim=1) == targets).float().mean().cpu())

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

    avg_val_loss = np.mean(val_loss)
    avg_val_accuracy = np.mean([a.item() for a in val_accuracy])  # Move each tensor to CPU and get the item

    return avg_val_loss, avg_val_accuracy


# Putting everything together
train_loader, val_loader, model, loss_function, optimizer, scheduler = get_train_setup(train_df, val_df)

# Training the model and saving the best model
model_output_path = "./cartographic_analysis/best_model.pth"  # Path to save the best model
train_values, best_val_loss, best_val_accuracy = train(model, model_output_path, loss_function, optimizer, scheduler, train_loader, val_loader, EPOCHS, evaluation=True)

# Save additional training results or logs if needed
# For example, you can save train_values to a CSV file or any other format.
torch.save(train_values, "./cartographic_analysis/train_values.pth")  # Save training values
