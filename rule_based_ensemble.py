from openai import OpenAI
import os
import argparse 
import pandas as pd
import json
import re
from collections import Counter
from utils import ChatClient, RetrievalAugmentedPrompt

if __name__ == "__main__":
    """System 1 generates the results for Task1 and Task2. It then finds relevant sentence from System2.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_loc", type=str, required=True,
                        help="Train file location")
    parser.add_argument("--val_loc", type=str, required=True,
                        help="Validation file location")
    parser.add_argument("--model_name", required=True,type=str,
                        help="Input the GPT model name")
    parser.add_argument("--api_key_f", required=True,type=str,
                        help="Input the API KEY for GPT-based models")
    parser.add_argument("--similarity_file_loc", required=True,type=str,
                        help="Input the Similarity file location. Similarity file contains similar texts from the MSTraining set")
    parser.add_argument("--reason_file_loc", required=True,type=str,
                        help="Input the reason file location collected from GPT4.")
    parser.add_argument("--system1_loc", type=str, required=True,
                        help="System 1 text file")
    parser.add_argument("--system2_loc", type=str, required=True,
                        help="System 2 GPT4 output file")
    parser.add_argument("--pass_number", type=int , required=True,
                        help="Which run of system 2 you want to use?")
    parser.add_argument("--output_file_loc", type=str, required=True,
                        help="Run file location" )
    
    
    args = parser.parse_args()
    rag =RetrievalAugmentedPrompt(args.train_loc, args.similarity_file_loc, args.reason_file_loc)
    model_name=args.model_name
    api_key=args.api_key_f
    chat_client=ChatClient(api_key_f=api_key, model=model_name)
    """GPT calls end
    """
    test_file_loc=args.val_loc
    encoding = 'latin1'
    test_df=pd.read_csv(test_file_loc, encoding=encoding)
    system1_data=[] #System 1 is Task 1 and Task 2 results from Zaholong
    with open(args.system1_loc, 'r') as file:
        for line in file:
            parts = line.split(' ', 3)
            system1_data.append(parts)
    system1_df = pd.DataFrame(system1_data, columns=['Text ID', 'Error Flag', 'Error Sentence ID', 'Corrected Sentence'])
    pass_number=args.pass_number
    pattern1 = r"Error Flag: (\d+)"
    pattern2=r"Error Sentence ID: (-?\d+)"
    pattern3=r"Corrected Sentence: (.+)"
    #System 2 is the RAG system
    with open(args.system2_loc, "r") as f:
        system2_dict=json.load(f)
    ensembled_result=[]
    number_of_corrections=0
    number_of_corrections_needed=0
    for index, row in system1_df.iterrows():
        text_id=row['Text ID']
        system1_error_flag=row['Error Flag']
        system1_error_sentence=row['Error Sentence ID']
        result={"Text ID":text_id,
                "Error Flag":system1_error_flag, 
                "Error Sentence ID":system1_error_sentence,
                "Corrected Sentence":row["Corrected Sentence"].strip()
                }
        if not int(system1_error_flag)==0:
            system2_pred=system2_dict[text_id]
            r1=[]
            r2=[]
            r3=[]
            for run in system2_pred:
                text = system2_pred[run]
                # Find all occurrences of the pattern in the text
                #print(text)
                matches1 = re.findall(pattern1, text)
                matches2=re.findall(pattern2, text)
                matches3=re.findall(pattern3, text)
                try:
                    r1.append(matches1[0])
                    r2.append(matches2[0])
                    r3.append(matches3[0])
                except:
                    continue
            try:
                system2_error_flag=r1[pass_number]
                system2_error_sentence=r2[pass_number]
                if int(system1_error_sentence)==int(system2_error_sentence):
                    result["Corrected Sentence"]=r3[pass_number]
                    number_of_corrections+=1
                else:
                    number_of_corrections_needed+=1
                    #matching_rows = test_df[test_df['Text ID'] == text_id]
                    sentences = test_df.loc[test_df['Text ID'] == text_id, 'Sentences'].iloc[0]
                    print(sentences)
                    q=rag.create_prompt(text_id, sentences, system1_error_sentence, neg=True)
                    print(q)
                    response=chat_client.get_response(q, neg=True)
                    print(response)
                    matches4=re.findall(pattern3, response)
                    result["Corrected Sentence"]=matches4[0]
                    input("Enter::")
                    break
            except:
                continue

        ensembled_result.append(result)
    with open(args.output_file_loc, "w") as f:
        for elem in ensembled_result:
            elem_list=[elem[k] for k in elem]
            line = ' '.join(map(str, elem_list))    
            f.write(f"{line}\n")
    print("Total number of corrections performed using GPT data files: ", number_of_corrections)
    print("Total number of corrections perfromed using GPT connection:", number_of_corrections_needed)
