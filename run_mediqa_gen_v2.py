from openai import OpenAI
import json
import asyncio
import pandas as pd
import numpy as np
import argparse
from utils import ChatClient, RetrievalAugmentedPrompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_loc", type=str, required=True,
                        help="Train file location")
    parser.add_argument("--val_loc", type=str, required=True,
                        help="Validation file location")
    parser.add_argument("--model_name", required=True,type=str,
                        help="Input the GPT model name")
    parser.add_argument("--api_key", required=True,type=str,
                        help="Input the API KEY for GPT-based models")
    parser.add_argument("--similarity_file_loc", required=True,type=str,
                        help="Input the Similarity file location. Similarity file contains similar texts from the MSTraining set")
    parser.add_argument("--reason_file_loc", required=True,type=str,
                        help="Input the reason file location collected from GPT4.")
    parser.add_argument("--output_file_loc", required=True,type=str,
                        help="Input the output file location.")
    args = parser.parse_args()
    
    args = parser.parse_args()
    rag =RetrievalAugmentedPrompt(args.train_loc, args.similarity_file_loc, args.reason_file_loc)
    model_name=args.model_name
    api_key=args.api_key
    chat_client=ChatClient(api_key_f=api_key, model=model_name)
    encoding = 'latin1'
    df=pd.read_csv(args.val_loc, encoding=encoding)
    answers={}
    output_file_loc=args.output_file_loc
    for index, row in df.iterrows():
        val_text_id=row['Text ID']
        answers[val_text_id]={}
        for run in range(1,4):
            sentences=row['Sentences']
            q=rag.create_prompt_for_rag(val_text_id, sentences, neg=False)
            print(q)
            input("Enter::")
            response=chat_client.get_response(q, neg=False)
            print(response)
            answers[val_text_id][run]=response
            with open(output_file_loc, "w") as f:
                json.dump(answers,f)
        break