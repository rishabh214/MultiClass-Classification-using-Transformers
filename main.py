from fastapi import FastAPI, Response, Depends, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import random
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
import pandas as pd

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)

class Request(BaseModel): 
    text : str


nltk.download('stopwords')
nltk.download('punkt')


def remove_stopwords(text):

    stopWords = set(stopwords.words('english'))
    try:
        words = word_tokenize(text)
    except Exception as e:
        print("### ",e,text,type(text))

    wordsFiltered = []

    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    text_filtered=' '.join(wordsFiltered)
    return text_filtered



def run_pred(device,model,dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
    
def test(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
    print("!!!!!!!!!!!!!!!")
    encoded_data_test = tokenizer.batch_encode_plus(
                                                df.desc.values, 
                                                add_special_tokens=True, 
                                                return_attention_mask=True, 
                                                pad_to_max_length=True, 
                                                max_length=256, 
                                                return_tensors='pt' )


    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(df.Class.values)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=8)
    
    return dataloader_test
  

def run(dataloader_test):

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=46,
                                                        output_attentions=False,
                                                        output_hidden_states=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.load_state_dict(torch.load('finetuned_BERT_epoch_3.model', map_location=torch.device('cpu')))

    _, predictions, true_vals = run_pred(device,model,dataloader_test)
    return _, predictions, true_vals



def driver2(para):
    
    df = pd.DataFrame()
    df['desc'] =[para] 
    df['Class']=[9]

    dataset=test(df)

    _, predictions, true_vals = run(dataset)

    return predictions,df


@app.post("/predict")
async def run_main(request:Request):

    text=request.text
    print("text",text)

    p,d=driver2(text)

    p1=np.argmax(p[0])

    p[0][p1]=-1

    p2=np.argmax(p[0])
    #print(p2)

    return {"Text Entered": text, "Predicted Class":str(p1),"Secondry Predicted Class":str(p2)}
