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
EPOCHS = 1
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
    


test_params = {
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 0
}

def calc_accuracy(preds, targets):
    accuracy = (preds==targets).cpu().numpy().mean() * 100
    return accuracy

def predict(model, loss_fn, dataloader):
    model.eval()

    test_accuracy = []
    test_loss = []
    test_preds = []

    for batch in dataloader:
        input_ids = batch['ids'].to(device, dtype = torch.long)
        attention_mask = batch['masks'].to(device, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.long)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)

        loss = loss_fn(outputs, targets)
        test_loss.append(loss.item())

        _, max_ids = torch.max(outputs.data, dim=1)
        test_accuracy.append(calc_accuracy(max_ids, targets))
        test_preds.append([max_ids.cpu().detach().numpy().reshape(-1), targets.cpu().detach().numpy().reshape(-1)])

    test_loss = np.mean(test_loss)
    test_accuracy = np.mean(test_accuracy)

    return test_loss, test_accuracy, test_preds



loss_function = torch.nn.CrossEntropyLoss()

     



test_data = TextDataset(test_df, tokenizer, MAX_LEN)
test_loader = DataLoader(test_data, **test_params)

print('test_data')
print(len(test_data))

model = HateSpeechClassifier(MODEL_NAME, 2)
state_dict = torch.load("./cartographic_analysis/best_model.pth", map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)


test_loss, test_accuracy, test_preds = predict(model, loss_function, test_loader)
test_loss, test_accuracy


preds = []
targets = []
for ele in test_preds:
    for p in ele[0]: preds.append(p)
    for p in ele[1]: targets.append(p)

preds_np = np.array(preds)
targets_np = np.array(targets)
print(preds_np.shape, targets_np.shape)

print(f1_score(targets_np, preds_np, average="weighted"))


print(classification_report(targets_np, preds_np,))


def get_pred(label_0, label_1):
    if label_0 > label_1:
        return 0
    else:
        return 1


train_values = torch.load("./cartographic_analysis/train_values.pth")

# Step 2: Print the loaded tensor
print("Loaded Tensor:")
print(train_values)



# Define the function to build the DataFrame
def build_train_values_df(train_values):
    train_values_temp = [0] * len(train_values)
    
    # Convert tensors to NumPy arrays
    for i, v in enumerate(train_values):
        train_values_temp[i] = v.cpu().detach().numpy()

    train_values_temp2 = []
    
    # Create a temporary array with additional information
    for i, v in enumerate(train_values_temp):
        for j, row in tqdm(enumerate(v), desc="Processing Rows"):
            row1 = np.append(row, i + 1)  # Append the epoch number (i+1)
            train_values_temp2.append(row1)

    # Stack the temporary array into a NumPy array
    train_values_np = np.stack(train_values_temp2, axis=0)

    # Create a DataFrame with the desired column names
    train_values_df = pd.DataFrame(train_values_np, columns=['text_id', 'label_0', 'label_1', 'epoch_no'])

    # Convert appropriate columns to integer types
    train_values_df['text_id'] = train_values_df['text_id'].astype(int)
    train_values_df['epoch_no'] = train_values_df['epoch_no'].astype(int)

    # Merge with another DataFrame (train_df)
    train_values_df = pd.merge(train_values_df, train_df[['text_id', 'text', 'label', 'agreement_level', 'agreement_factor']],
                                on='text_id', how='left')

    # Apply prediction logic
    train_values_df['pred'] = train_values_df.apply(lambda row: get_pred(row.label_0, row.label_1), axis=1)

    return train_values_df

# Build the DataFrame
train_values_df = build_train_values_df(train_values)

# Display the first few rows of the resulting DataFrame
print(train_values_df.head())






# Define the get_pred function based on your requirements
def get_pred(label_0, label_1):
    return 0 if label_0 > label_1 else 1  # Example logic for prediction

# Define the confidence, variability, and correctness functions
def get_confidence(label, label_0_mean, label_1_mean):
    return label_0_mean if label == 0 else label_1_mean

def get_variability(label, label_0_std, label_1_std):
    return label_0_std if label == 0 else label_1_std

def get_correctness(label, label_0_last, label_1_last):
    return round(label_0_last * 5) / 5 if label == 0 else round(label_1_last * 5) / 5



# Define metrics for aggregation
metrics = ['mean', 'std', 'last']

# Define the function to build the cartography DataFrame
def build_cartography_df(train_values_df):
    # Aggregate metrics
    agg_df = train_values_df.sort_values(['epoch_no']).groupby('text_id', as_index=False).agg({
        'text': 'first',
        'label_0': metrics,
        'label_1': metrics,
        'label': 'first',
        'pred': 'last',
        'agreement_level': 'first',
        'agreement_factor': 'first'
    })
    
    # Rename columns to include aggregated metrics
    agg_df.columns = [
        'text_id', 'text', 
        'label_0_mean', 'label_0_std', 'label_0_last', 
        'label_1_mean', 'label_1_std', 'label_1_last', 
        'label', 'pred', 
        'agreement_level', 'agreement_factor'
    ]

    # Calculate confidence, variability, and correctness
    agg_df['confidence'] = agg_df.apply(
        lambda row: get_confidence(row.label, row.label_0_mean, row.label_1_mean), axis=1
    )
    agg_df['variability'] = agg_df.apply(
        lambda row: get_variability(row.label, row.label_0_std, row.label_1_std), axis=1
    )
    agg_df['correctness'] = agg_df.apply(
        lambda row: get_correctness(row.label, row.label_0_last, row.label_1_last), axis=1
    )
    agg_df['is_correct'] = np.where((agg_df['label'] == agg_df['pred']), True, False)

    return agg_df

# Build the cartography DataFrame
cartography_df = build_cartography_df(train_values_df)

# Display the first few rows of the resulting cartography DataFrame
print(cartography_df.head())


sns.scatterplot(data=cartography_df, x="variability", y="confidence", hue='correctness', palette='flare')
plt.xlim(-0.1, 0.5)  # Set limits based on your expectations
plt.ylim(0.3, 1.0)   # Adjust as necessary
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='correctness')
plt.savefig("./cartographic_analysis/conf_vs_var_color_correctness_final_2.png", dpi=600, bbox_inches='tight')

