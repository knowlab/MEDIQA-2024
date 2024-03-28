from openai import OpenAI
import json
import asyncio
import pandas as pd
import numpy as np
import argparse
def get_chat_response(client:object,
                      model:str,
                      system_message: str, 
                      user_request: str, 
                      seed: int = None, 
                      temperature: float = 0.7):
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_request},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed,
            max_tokens=200,
            temperature=temperature,
        )

        response_content = response.choices[0].message.content
        return response_content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def collect_ICL_examples(val_text_id, train_file_loc, similarity_dict, reason_file):
    df= pd.read_csv(train_file_loc)
    sim_dict=similarity_dict[val_text_id]
    sim_list=[key for key in sim_dict]
    df_slected = df[df['Text ID'].isin(sim_list)]
    error_flag_0 = df_slected[df_slected['Error Flag'] == 0]
    error_flag_1 = df_slected[df_slected['Error Flag'] == 1]
    # Randomly select 2 rows where "error_flag" is 0
    selected_rows_0 = error_flag_0.sample(n=2, random_state=42)
    # Randomly select 2 rows where "error_flag" is 1
    selected_rows_1 = error_flag_1.sample(n=2, random_state=42)

    # Concatenate the selected rows into a new DataFrame
    new_df = pd.concat([selected_rows_0, selected_rows_1])
    shuffled_df = new_df.sample(frac=1, random_state=42)
    with open(reason_file, "r") as f:
        reason_dict=json.load(f)
    prompt_template=''''''
    reason_template='''Reasons:
    '''
    idx=0
    for index, row in shuffled_df.iterrows():
        idx+=1
        text_id=row['Text ID']
        text_msg=row["Sentences"]
        error_flag=row["Error Flag"]
        error_sentence_no=row["Error Sentence ID"]
        error_sentence=row["Error Sentence"]
        corrected_sentence=row["Corrected Sentence"]
        prompt_template+="Example: " +str(idx)+"\n"
        prompt_template+=text_msg
        prompt_template+="\nOutput:\n"
        if error_flag==0:
            being_correct="contains no error"
        else:
            being_correct="contains an error"
        reason_template+="Example " +str(idx)+" "+being_correct+"  because:\n "+reason_dict[text_id]+"\n"
        prompt_template+="Error Flag: "+ str(error_flag)+"\n"
        prompt_template+="Error Sentence ID: "+ str(error_sentence_no)+"\n"
        if error_flag==0:
            prompt_template+="Corrected Sentence: NA"+"\n\n"
        else:
            prompt_template+="Corrected Sentence:"+corrected_sentence+"\n\n"
    return prompt_template, reason_template
def get_system_msg():
    system_msg='''Role: you are a helpful clinical assistant who has clinical knowledge. 
        You will be provided with four Examples of patient records, each of which may or may not contain an error. 
        I will specify reasons for each record being correct under the "Reason" section. 
        Then, you will be given a "Patient Record" to analyze whether it contains a medical error. 
        You should examine the diagnosis, management, and intervention-related statements.
        If an error is found in the Patient Record, set the Error Flag to 1, output the Error Sentence ID, 
        and generate the Corrected Sentence. Otherwise, set the Error Flag to 0, Error Sentence ID to -1, and Corrected Sentence to "NA". \n
        Please refer to the Examples for the expected output format and understand the Reasons for the Examples being correct or not.'''
    return system_msg


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
    train_file_loc=args.train_loc
    validation_file_loc=args.val_loc
    model_name=args.model_name
    api_key_f=args.api_key
    similarity_file_name=args.similarity_file_loc
    output_file_loc=args.output_file_loc
    reason_file_loc=args.reason_file_loc
    gpt_client = OpenAI(api_key=api_key_f)
    def get_response(user_request):
        system_msg=get_system_msg()
        response = get_chat_response(
            client=gpt_client, model=model_name, system_message=system_msg, user_request=user_request
        )
        return response
    
    with open(similarity_file_name, "r") as f:
        similarity_dict=json.load(f)

    encoding = 'latin1'
    df=pd.read_csv(validation_file_loc, encoding=encoding)
    answers={}
    for index, row in df.iterrows():
        val_text_id=row['Text ID']
        answers[val_text_id]={}
        for run in range(1,4):
            icl_examples, reasons=collect_ICL_examples(val_text_id,train_file_loc, similarity_dict, reason_file_loc)
            output_template=''''''
            output_template+=reasons
            output_template+=''' 
            Patient Record:\n'''
            output_template+=row['Sentences']
            output_template+='''
            \n Output format:
            \nError Flag:<number>
            \nError Sentence ID:<number>
            \n Corrected Sentence:<text>
            '''
            q="".join([icl_examples, output_template])
            #q=icl_examples+output_template
            response=get_response(q)
            print(response)
            answers[val_text_id][run]=response
            with open(output_file_loc, "w") as f:
                json.dump(answers,f)
    