# text2sql
T5 Text-to-SQL Conversion
This project utilizes the T5 base model for the task of converting natural language queries into SQL queries. Fine-tuning of the T5 model is conducted with adjustments to various hyperparameters using values specified in JSON configuration files.

Training Process
Fine-Tuning Parameters: All parameters of the T5 base model are fine-tuned during training.
Epochs: Training is conducted over 20 epochs.
Early Stopping: Early stopping is applied using validation and training loss to prevent overfitting.
Evaluation Metrics: During training, the model is evaluated using ROUGE and BLEU metrics.
Main Evaluation Metrics: For testing, ROUGE and BLEU metrics are primarily employed, along with query accuracy and query execution accuracy.
Model Performance
The chosen model achieved the following results:

ROUGE Scores:
rouge1: 0.9601
rouge2: 0.9212
rougeL: 0.9459
rougeLsum: 0.9461
BLEU Scores:
bleu: 0.8971
precisions:
0.9549
0.9217
Test Loss: 0.0573
Execution Accuracy: 0.7778
Query Accuracy: 0.7094
These metrics highlight the effectiveness of the model in accurately converting natural language queries to SQL queries, showcasing its potential for practical applications in database interaction and information retrieval tasks.
