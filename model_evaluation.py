import sqlite3
import torch
from data_loading import InputTextDataset
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
import os
import pandas as pd
import matplotlib.pyplot as plt

def execute_query(db_path: str, query: str) -> list:
    """
    Execute the provided SQL query on the specified SQLite database.

    Args:
    - db_path (str): Path to the SQLite database file.
    - query (str): SQL query to execute.
    Returns:
    - result (list): List of rows returned by the query execution, or -1 if an error occurs.
    """
    result = []
    
    try:
        # Connect to the SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        # Execute the query
        cursor.execute(query)

        result = cursor.fetchall()
        connection.commit()
        connection.close()
    except sqlite3.Error as e:

        return -1
    return result

def compare_query_execution_results(db_path: str, predicted_query: str, reference_query: str) -> bool:
    """
    Compare the results of two SQL queries executed on the specified SQLite database.

    Args:
    - db_path (str): Path to the SQLite database file.
    - predicted_query (str): First SQL query to execute.
    - reference_query (str): Second SQL query to execute.
    """
    if compute_query_accuracy(predicted_query, reference_query):
        return 1 #similiar case
        
    # Execute both queries
    result1 = execute_query(db_path, reference_query)
    result2 = execute_query(db_path, predicted_query)
    
    if result1 == -1 or result2 == -1:
        return 0
    # Compare results
    similarity = int(result1 == result2)
    return similarity


def get_sql_code(query_with_prefix, prefix = 'Generated SQL code:'):
    """
    Extract the SQL code from a query string that starts with a prefix.
    
    Arguments:
    query_with_prefix (str): Query string that starts with a prefix, e.g., 'Generated SQL: SELECT * FROM table'.
    prefix (str): prefix added to the input.

    """
    
    # Check if the query starts with the prefix
    if query_with_prefix.startswith(prefix):
        return query_with_prefix[len(prefix):].strip()
    else:
        # If no prefix is found, return the original string
        return query_with_prefix.strip()

def visualize_training_metrics(model_dir):
    df = pd.read_csv(os.path.join(model_dir, "Metrics.csv"))
    
    # Number of epochs
    num_epochs = len(df)
  
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Metrics to plot
    metrics_to_plot = [
        ('train_loss', 'val_loss'), 'val_rouge', 'val_bleu'
    ]
    
    titles = [
        'Training and Validation Loss', 'Validation Rouge Score', 'Validation Bleu Score'
    ]
    
    for i, metric_pair in enumerate(metrics_to_plot):
        ax = axes[i]
        
        if isinstance(metric_pair, tuple):  
            train_metric, val_metric = metric_pair
            ax.plot(range(1, num_epochs + 1), df[train_metric], label='Train')
            ax.plot(range(1, num_epochs + 1), df[val_metric], label='Validation')
            ax.set_ylabel('Value')
            ax.legend()
        else:  
            metric = metric_pair          
            ax.plot(range(1, num_epochs + 1), df[metric], label='Validation')
            ax.set_ylabel('Value')
            ax.legend()
        
        ax.set_xlabel('Epoch')
        ax.set_title(titles[i])  # Set the title based on the metric name
    plt.tight_layout()
    image_path = os.path.join(model_dir, "metrics.jpg")
    plt.savefig(image_path)
    plt.show()

def load_model(model_checkpoint_path):
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
    model_state_dict = torch.load(model_checkpoint_path)['model']
    model.load_state_dict(model_state_dict)
    model.eval()
    return model      

def get_test_evaluation_metrics(df_path='text-to-sql_from_spider.csv', model_checkpoint_path =  'fine_tuning_text_sql_model/best_loss_t5_base_model.pth',
spider_db_dir = 'database_folder/spider_databases/spider/database/', device= 'cpu'):
    
    """
    Evaluate the performance of the model on the test dataset using various metrics.

    Args:
    - df_path (str): Path to the CSV file containing the test dataset.
    - model_checkpoint_path (str): Path to the model checkpoint file.
    - spider_db_dir (str): Directory containing the Spider databases.
    - device (str): Device on which to run the evaluation (e.g., 'cpu', 'cuda').

    Returns:
    - dict: A dictionary containing evaluation metrics including Rouge scores, BLEU scores,
            test loss, execution accuracy, and query accuracy.
    """
    
    sql_df = pd.read_csv(df_path)
    train, test = train_test_split(sql_df, test_size=0.2, random_state=42, shuffle=True)
    val, test = train_test_split(test, test_size=0.5, random_state=42, shuffle=True)
    test = test.reset_index()

    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
    
    test_dataset = InputTextDataset(test,tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_model(model_checkpoint_path, model)
    model.to(device)

    rouge_metric =  evaluate.load('rouge')
    bleu_metric = evaluate.load("bleu")
    batch_test_loss = []
    for test_data, test_target, test_attention_mask, spider_db_name in test_loader:
        test_data, test_target, test_attention_mask = test_data.to(device).squeeze().unsqueeze(0), test_target.to(device).squeeze().unsqueeze(0), test_attention_mask.to(device).squeeze().unsqueeze(0)
        test_output = model.generate(test_data, max_new_tokens = 2000)
        decoded_test_target = [tokenizer.decode(test_target[0][test_target[0] != -100], skip_special_tokens=True)]
        test_result = tokenizer.batch_decode(test_output, skip_special_tokens=True)
        batch_test_loss.append( model(input_ids=test_data, attention_mask=test_attention_mask, labels=test_target).loss.item())
    
        # Compute Rouge scores
        rouge_metric.add_batch(predictions= test_result, references= decoded_test_target)
    
        # Compute BLEU scores
        bleu_metric.add_batch(predictions= test_result, references= decoded_test_target)

        gold_query = get_sql_code(decoded_test_target[0])
        predicted_query = get_sql_code(test_result[0])
        spider_sqlite_path = f'{spider_db_dir}/{spider_db_name[0]}/{spider_db_name[0]}.sqlite'
      
    queries_excecution_comparision.append(compare_query_execution_results(db_path = spider_sqlite_path ,predicted_query = predicted_query, reference_query = gold_query))
    queries_comparision.append(compute_query_accuracy(predicted_query = predicted_query, reference_query = gold_query))
    
    rouge_scores = rouge_metric.compute()
    bleu_scores = bleu_metric.compute()
    test_loss = sum(batch_test_loss)/ len(batch_test_loss)
    execution_accuracy = sum(queries_excecution_comparision)/len(queries_excecution_comparision)
    query_accuracy =  sum(queries_comparision)/len(queries_comparision)

    return {
            "rouge_scores": rouge_scores,
            "bleu_scores": bleu_scores,
            "test_loss": test_loss,
            "execution_accuracy": execution_accuracy,
            "query_accuracy": query_accuracy
        }
