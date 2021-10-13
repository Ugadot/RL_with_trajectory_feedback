from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 160

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        pooled_output = bert_out.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

    def get_sentiment(self, text):
        device = self.bert.device
        encoded_text = self.tokenizer.encode_plus(
                      text,
                      max_length=MAX_LEN,
                      add_special_tokens=True,
                      return_token_type_ids=False,
                      pad_to_max_length=True,
                      return_attention_mask=True,
                      return_tensors='pt',
                    )
        
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        output = self.forward(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

        # print(f'Review text: {review_text}')
        # print(f'Sentiment  : {class_names[prediction]}')

        return output

def create_sentiment_classifier(pretrained=True):
    model = SentimentClassifier(n_classes=3)
    if pretrained:
        sd = torch.load('data/nlp/bert_sa.pt')
        model.load_state_dict(sd)

    model.eval()    
    return model

if __name__ == '__main__':
    classifier = create_sentiment_classifier()
    text1 = "you did great!"
    text2 = "that was terrible"
    print(text1, classifier.get_sentiment(text1))
    print(text2, classifier.get_sentiment(text2))
