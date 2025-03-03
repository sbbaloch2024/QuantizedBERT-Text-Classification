import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Define the IMDbDataset Class
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 2. Load and Preprocess the IMDB Dataset
def load_and_preprocess_data(tokenizer, max_length=512, test_size=0.1):
    # Load the dataset
    dataset = load_dataset("imdb")

    # Split the dataset into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataset['train']['text'], dataset['train']['label'], test_size=test_size, random_state=42
    )

    # Tokenize the data
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=max_length)

    # Create PyTorch Datasets
    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)

    return train_dataset, val_dataset

# 3. Fine-Tune the TinyBERT Model
def fine_tune_model(train_dataset, val_dataset, model_name="huawei-noah/TinyBERT_General_4L_312D", num_labels=2, output_dir='./results', num_epochs=1, batch_size=16, learning_rate=2e-5):
    # Load the pre-trained model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Align save strategy with evaluation strategy
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the fine-tuned model using Hugging Face's method
    trainer.save_model("fine_tuned_tinybert")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained("fine_tuned_tinybert")

    return model

# 4. Apply Dynamic Quantization
def apply_dynamic_quantization(model, dtype=torch.qint8):
    # Apply dynamic quantization on Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=dtype
    )

    # Why are we using torch here  
    return quantized_model

# 5. Save the Quantized Model's State Dict
def save_quantized_model_state_dict(quantized_model, path="quantized_tinybert_state_dict.pth"):
    # Save only the state dict of the quantized model
    torch.save(quantized_model.state_dict(), path)
    print(f"Quantized model state dict saved to {path}")

# 6. Load the Quantized Model's State Dict and Apply Quantization
def load_quantized_model(model_dir="fine_tuned_tinybert", state_dict_path="quantized_tinybert_state_dict.pth", num_labels=2):
    # Load the fine-tuned model architecture
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    
    # Load the state dict
    state_dict = torch.load(state_dict_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_model.eval()
    print("Quantized model loaded and moved to CPU.")
    return quantized_model

# 7. Perform Inference with the Quantized Model
def perform_inference(quantized_model, tokenizer, test_texts, max_length=512):
    # Tokenize the test texts
    test_encodings = tokenizer(
        test_texts, truncation=True, padding='max_length', return_tensors='pt', max_length=max_length
    )

    # Move inputs to CPU
    input_ids = test_encodings['input_ids']
    print(type(input_ids), input_ids.shape)  # Should be a torch.Tensor
    attention_mask = test_encodings['attention_mask']
    print(type(attention_mask), attention_mask.shape)  # Should be a torch.Tensor
    # token_type_ids is optional; remove if not needed
    # token_type_ids = test_encodings.get('token_type_ids', torch.zeros_like(test_encodings['input_ids']))
    # print(type(token_type_ids), token_type_ids.shape)  # Should be a torch.Tensor

    # Ensure the model is on CPU
    device = torch.device('cpu')
    quantized_model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    # token_type_ids = token_type_ids.to(device)  # Uncomment if needed

    # Perform inference
    with torch.no_grad():
        outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        print("Predicted labels:", predictions)

# 8. Evaluate the Quantized Model on the Validation Set
def evaluate_quantized_model(quantized_model, val_dataset, batch_size=16):
    # Create DataLoader for validation
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    quantized_model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print(f"Quantized Model Evaluation Results:")
    print(f" - Average Loss: {avg_loss:.4f}")
    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall: {recall:.4f}")
    print(f" - F1 Score: {f1:.4f}")

# 9. Main Function to Execute All Steps
def main():
    # Define parameters
    MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
    NUM_LABELS = 2
    OUTPUT_DIR = './results'
    NUM_EPOCHS = 1
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512
    QUANTIZED_MODEL_PATH = "quantized_tinybert_state_dict.pth"
    TEST_TEXTS = ["This movie was fantastic!", "I did not like this film."]

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Load and preprocess data
    train_dataset, val_dataset = load_and_preprocess_data(tokenizer, max_length=MAX_LENGTH, test_size=0.1)

    # Fine-tune the model
    print("Starting model fine-tuning...")
    fine_tuned_model = fine_tune_model(
        train_dataset, val_dataset,
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    # Apply dynamic quantization
    print("Applying dynamic quantization...")
    quantized_model = apply_dynamic_quantization(fine_tuned_model, dtype=torch.qint8)

    # Save the quantized model's state dict
    save_quantized_model_state_dict(quantized_model, path=QUANTIZED_MODEL_PATH)

    # Load the quantized model for inference and evaluation
    print("Loading the quantized model for inference and evaluation...")
    quantized_model_loaded = load_quantized_model(
        model_dir="fine_tuned_tinybert",
        state_dict_path=QUANTIZED_MODEL_PATH,
        num_labels=NUM_LABELS
    )

    # Perform inference
    print("Performing inference with the quantized model...")
    perform_inference(quantized_model_loaded, tokenizer, TEST_TEXTS, max_length=MAX_LENGTH)

    # Evaluate the quantized model on the validation set
    print("Evaluating the quantized model on the validation set...")
    evaluate_quantized_model(quantized_model_loaded, val_dataset, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()
