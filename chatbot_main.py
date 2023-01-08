import numpy as np
import pandas as pd
import torch
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import os

device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "/Users/jeeho/lab/chatbot/static/"
print(PATH)
      
A = '정환'
B = '택'
C = '선우'
D = '동룡'

jh_model = torch.load(PATH + A +'.pt', map_location=device)
t_model = torch.load(PATH + B +'.pt', map_location=device)
sw_model = torch.load(PATH + C +'.pt', map_location=device)
dr_model = torch.load(PATH + D +'.pt', map_location=device)


tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",bos_token='</s>', eos_token='</s>', unk_token='<unk>',pad_token='<pad>', mask_token='<mask>')


def choose_character():
    
    who = input("두근두근 소개팅 대화 상대를 골라주세요 >< > \n(정환,택,선우,동룡 중 1명 입력)").strip()
    
    while 1:
      if who =='정환':
        model = jh_model
        break
      if who =='택':
        model = t_model
        break
      if who =='선우':
        model = sw_model
        break
      if who =='동룡':
        model = dr_model
        break
      else : 
        who = input("이 중에서는 맘에 드는 상대가 없으신가요? ㅠㅠ > \n(정환,택,선우,동룡 중 1명 입력)").strip()
        
    user = input("본인 이름을 입력해주세요! > ").strip()
    return who, user, model
    


def chat(who, user, model):
    
    
    print("끼이익 또각또각 쿵.....")
    print(who+" > 안녕 {}!".format(user.strip()))
    
    
    with torch.no_grad():    
        while 1:
            q = input(user+" > ").strip()
            if q == "우리 그만하자.":
                break
            a = ""
            
            while 1:
                input_ids = torch.LongTensor(tokenizer.encode("<usr>" + q +"<sys>" + a)).unsqueeze(dim=0)
                input_ids.to(device)
                pred = model(input_ids)
                pred = pred.logits
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                if gen == '</s>':
                    break
                a += gen.replace("▁", " ")
            print(who+" > {}".format(a.strip()))


who, user, model = choose_character()
print("지금부터 "+who+"의 소개팅이 시작됩니다. 두근두근! (우리 그만하자. 입력시 종료)" )
chat(who, user, model)