import pandas as pd
import numpy as np
import os
import argparse
import json
from data_loading import InputTextDataset
from sklearn.model_selection import train_test_split
import torch
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from pathlib import Path
from torch.utils.data import Dataset, DataLoader



# Define your training and validation loop
def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, run_directory):
    print(run_directory)
    if not os.path.exists(run_directory):
        os.makedirs(run_directory)

# Save model information to a JSON file
        model_info = {
        "model": "T5_base",
        "batch_size": str(args.batch_size),  
        "fine_tune_type": "training all parameters",
        "learning_rate": str(args.learning_rate),
        "adam_epsilon": str(args.adam_epsilon), 
        "weight_decay": str(args.weight_decay),  
        "optimizer": "AdamW"
    }

    with open(os.path.join(run_directory, "model_info.json"), "w") as info_file:
        json.dump(model_info, info_file)
        
    # Move model to device
    model.to(device)

    column_names = [ 'val_rouge', 'val_bleu', 'val_loss', 'train_loss']

    metrics_df = pd.DataFrame(columns=column_names)
        
    best_model = None
    best_model_bleu = -1
    best_model_rouge = -1
    best_model_loss = 100
    train_loss_epochs = []
    val_loss_epochs = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        batch_train_losses = []
        model.train()
        for train_data, train_target, train_attention_mask, spider_db_name in train_loader:
            train_data, train_target, train_attention_mask = train_data.to(device).squeeze(), train_target.to(device).squeeze(), train_attention_mask.to(device).squeeze()
            optimizer.zero_grad()
            output = model.generate(train_data, max_new_tokens = 2000)
            loss = (model(input_ids=train_data, attention_mask=train_attention_mask, labels=train_target)).loss
            loss_value  = loss.item()
            batch_train_losses.append(loss_value)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        print('validation')
        train_epoch_loss = sum(batch_train_losses)/ len(batch_train_losses)
        train_loss_epochs.append(train_epoch_loss)
        
        # Save the entire model, model state_dict, optimizer sstate_dict, and the loss
        model_checkpoint = {
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,  # Save the current epoch
        }
        
        batch_val_loss = []
        with torch.no_grad():
            rouge_metric =  evaluate.load('rouge')
            bleu_metric = evaluate.load("bleu")
            for val_data, val_target, val_attention_mask, spider_db_name in val_loader:
                val_data, val_target, val_attention_mask = val_data.to(device).squeeze().unsqueeze(0), val_target.to(device).squeeze().unsqueeze(0), val_attention_mask.to(device).squeeze().unsqueeze(0)
                val_output = model.generate(val_data, max_new_tokens = 2000)
                decoded_val_target = [tokenizer.decode(val_target[0][val_target[0] != -100], skip_special_tokens=True)]
                val_result = tokenizer.batch_decode(val_output, skip_special_tokens=True)
                batch_val_loss.append( model(input_ids=val_data, attention_mask=val_attention_mask, labels=val_target).loss.item())
                # Compute Rouge scores
                rouge_metric.add_batch(predictions= val_result, references= decoded_val_target)
                # Compute BLEU scores
                bleu_metric.add_batch(predictions= val_result, references= decoded_val_target)

            rouge_scores = rouge_metric.compute()
            bleu_scores = bleu_metric.compute()
            
        epoch_val_loss = sum(batch_val_loss)/ len(batch_val_loss)
        val_loss_epochs.append(epoch_val_loss)
        
        metrics_df.loc[epoch, 'val_bleu']  = bleu_scores['bleu']
        metrics_df.loc[epoch, 'val_rouge']  =  rouge_scores['rouge1']
        metrics_df.loc[epoch, 'val_loss']  =  epoch_val_loss
        metrics_df.loc[epoch, 'train_loss']  =  train_epoch_loss

        if epoch_val_loss < best_model_loss:
            best_model_loss = epoch_val_loss
            torch.save(
            model_checkpoint,
            Path(run_directory) / f"best_loss_t5_base_model.pth",
        )
            print(f"Saved best model with loss metric at epoch {epoch}")

        if rouge_scores['rouge1'] > best_model_rouge:
            best_model_rouge = rouge_scores['rouge1']
            torch.save(
            model_checkpoint,
            Path(run_directory) / f"best_rouge_t5_base_model.pth",
        )
            print(f"Saved best rouge model at epoch {epoch}")

        if bleu_scores['bleu'] > best_model_bleu:
            best_model_bleu = bleu_scores['bleu']
            torch.save(
            model_checkpoint,
            Path(run_directory) / f"best_bleu_t5_base_model.pth",
        )
            print(f"Saved best bleu model at epoch {epoch}")
    
        metrics_df.to_csv(os.path.join(run_directory, "Metrics.csv"))
        
        torch.save(
                model_checkpoint,
                Path(run_directory) / f"checkpoint_epoch_{epoch}_t5_base_model.pth",
            )        
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for text-to-SQL task")

    parser.add_argument("--df_path", type=str, default="text-to-sql_from_spider.csv",
                        help="Path to the CSV file containing the dataset (default: text-to-sql_from_spider.csv)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for the optimizer (default: 5e-5)")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon parameter for AdamW optimizer (default: 1e-8)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay parameter for AdamW optimizer (default: 0.01)")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training DataLoader (default: 16)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for model training (default: cpu)")
    parser.add_argument("--run_directory", type=str, default="fine_tuning_text_sql_model",
                        help="Directory where models and metrics will be saved (default: fine_tuning_text_sql_model)")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    # Load dataset
    sql_df = pd.read_csv(args.df_path)
    train, test = train_test_split(sql_df, test_size=0.2, random_state=42, shuffle=True)
    val, test = train_test_split(test, test_size=0.5, random_state=42, shuffle=True)
    val = val.reset_index()
    train = train.reset_index()
    test = test.reset_index()
    
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")

    # Create AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)

    # Prepare DataLoader for training and validation
    train_dataset = InputTextDataset(train, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = InputTextDataset(val, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Train the model
    model = train_model(model, train_loader, val_loader, optimizer, num_epochs=args.num_epochs, device=args.device, run_directory = args.run_directory)
        
