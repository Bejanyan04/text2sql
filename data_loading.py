from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")

class InputTextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.input_text = []
        self.label_text = []
       

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_source_length = 512
        max_target_length = 128
  
        sample_input = "Write SQL query according to instruction: Insruction of the code to be generated: {} CONTEXT: {}".format(self.df.loc[idx,'question'], self.df.loc[idx, 'schema'])
        sample_label =  "Generated SQL code: {}".format(self.df.loc[idx,'sql'])
        
        input_encoding = tokenizer(
             sample_input,
              padding="max_length",
              max_length=max_source_length,
              truncation=True,
              return_tensors="pt",
          )
        
        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
      
          # encode the targets
        target_encoding = tokenizer(
              sample_label,
              padding="max_length",
              max_length=max_target_length,
              truncation=True,
              return_tensors="pt",
          )
        labels = target_encoding.input_ids
        
        labels[labels == tokenizer.pad_token_id] = -100
        spider_db_name = self.df.loc[idx, 'spider_db_name']
        
        return input_ids, labels, attention_mask, spider_db_name
        