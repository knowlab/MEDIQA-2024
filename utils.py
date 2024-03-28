import pandas as pd
import json
from openai import OpenAI
class ChatClient:
    def __init__(self, api_key_f,model):
        print("Invoking chat client")
        self.gpt_client = OpenAI(api_key=api_key_f)
        self.model=model
    
    def get_chat_response(self, client:object,
                      model:str,
                      system_message: str, 
                      user_request: str, 
                      seed: int = None, 
                      temperature: float = 0.7):
        try:
            print(system_message)
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
    
    def get_system_msg_for_ensemble(self, msg=None):
        if not msg:
            system_msg='''Role: you are a helpful clinical assistant who has clinical knowledge. 
                You will be provided with four Examples of patient records, each of which contains an error. 
                I will specify reasons for each record being incorrect under the "Reason" section. 
                Then, you will be given a "Patient Record" with a possible Error Sentence ID to analyze its error. 
                You should examine the diagnosis, management, and intervention-related statements.
                If an error is found in the Patient Record then generate a Corrected Sentence.'''
            return system_msg
        else:
            msg
    def get_system_msg(self, msg=None):
        if not msg:
            system_msg='''Role: you are a helpful clinical assistant who has clinical knowledge. 
        You will be provided with four Examples of patient records, each of which may or may not contain an error. 
        I will specify reasons for each record being correct under the "Reason" section. 
        Then, you will be given a "Patient Record" to analyze whether it contains a medical error. 
        You should examine the diagnosis, management, and intervention-related statements.
        If an error is found in the Patient Record, set the Error Flag to 1, output the Error Sentence ID, 
        and generate the Corrected Sentence. Otherwise, set the Error Flag to 0, Error Sentence ID to -1, and Corrected Sentence to "NA". \n
        Please refer to the Examples for the expected output format and understand the Reasons for the Examples being correct or not.'''
            return system_msg
        else:
            msg
    def get_response(self, user_request, neg=False):
        if neg:
            system_msg=self.get_system_msg_for_ensemble()
        else:
            system_msg=self.get_system_msg()
        response = self.get_chat_response(
            client=self.gpt_client, model=self.model, system_message=system_msg, user_request=user_request
        )
        return response

class RetrievalAugmentedPrompt:
    def __init__(self, train_file_loc, similarity_file_loc, reason_file_loc):
        """
        Parameters:
           train_file_loc (str): train_data location
           similarity_file_loc(str): similarity_data location
           reason_file_loc(str): reason_data location
        """
        self.train_data=pd.read_csv(train_file_loc)
        with open(similarity_file_loc, "r") as f:
            self.similarity_data=json.load(f)
        with open(reason_file_loc, "r") as f:
            self.reason_data=json.load(f)

    def example_template(self, df):
        idx=0
        example_template=''''''
        reason_template='''Reasons:
        '''
        for index, row in df.iterrows():
            idx+=1
            text_id=row['Text ID']
            text_msg=row["Sentences"]
            error_flag=row["Error Flag"]
            error_sentence_no=row["Error Sentence ID"]
            corrected_sentence=row["Corrected Sentence"]
            being_correct = "contains no error" if error_flag == 0 else "contains an error"
            example_template += f"Example: {idx}\n{text_msg}\nOutput:\n"
            reason_template += f"Example {idx} {being_correct} because:\n {self.reason_data[text_id]}\n"
            example_template += f"Error Flag: {error_flag}\nError Sentence ID: {error_sentence_no}\n"
            example_template += "Corrected Sentence: NA\n\n" if error_flag == 0 else f"Corrected Sentence:{corrected_sentence}\n\n"
        return example_template, reason_template
    
    def create_example_set(self,text_id):
        """
        Create exmple prompts with reason prompt
        Returns:
            tuple: prompt and reason template.
        """
        sim_dict=self.similarity_data[text_id]
        sim_list=[key for key in sim_dict]
        train_data=self.train_data
        df_slected = train_data[train_data['Text ID'].isin(sim_list)]
        error_flag_0 = df_slected[df_slected['Error Flag'] == 0]
        error_flag_1 = df_slected[df_slected['Error Flag'] == 1]
        # Randomly select 2 rows where "error_flag" is 0
        selected_rows_0 = error_flag_0.sample(n=2, random_state=42)
        # Randomly select 2 rows where "error_flag" is 1
        selected_rows_1 = error_flag_1.sample(n=2, random_state=42)

        # Concatenate the selected rows into a new DataFrame
        new_df = pd.concat([selected_rows_0, selected_rows_1])
        shuffled_df = new_df.sample(frac=1, random_state=42)
        ex, rs= self.example_template(shuffled_df)
        return ex, rs
   
    def create_negative_example_set(self,text_id):
        """
        Create exmple prompts with reason prompt
        Returns:
            tuple: prompt and reason template.
        """
        sim_dict=self.similarity_data[text_id]
        sim_list=[key for key in sim_dict]
        df=self.train_data
        df_slected = df[df['Text ID'].isin(sim_list)]
        error_flag_1 = df_slected[df_slected['Error Flag'] == 1] #only select negative examples 
        new_df = error_flag_1.sample(n=4, random_state=42)
        shuffled_df = new_df.sample(frac=1, random_state=42)
        ex, rs= self.example_template(shuffled_df)
        return ex, rs
    
    def create_prompt(self, text_id, sentences, sentence_id, neg=False):
        if neg:
            icl_examples, reasons=self.create_negative_example_set(text_id)
        else:
            icl_examples, reasons=self.create_example_set(text_id)    
        output_template=''''''
        output_template+=reasons
        output_template+=''' 
        Patient Record:\n'''
        output_template+=sentences
        output_template+=f'''
        \n Possible Error Sentence ID: {sentence_id}
        \n Output format:
        \n Corrected Sentence:<text>
        '''
        q="".join([icl_examples, output_template])
        return q
    
    def create_prompt_for_rag(self, text_id, sentences, neg=False):
        if neg:
            icl_examples, reasons=self.create_negative_example_set(text_id)
        else:
            icl_examples, reasons=self.create_example_set(text_id)    
        output_template=''''''
        output_template+=reasons
        output_template+=''' 
        Patient Record:\n'''
        output_template+=sentences
        output_template+=f'''
       \n Output format:
        \nError Flag:<number>
        \nError Sentence ID:<number>
        \n Corrected Sentence:<text>
        '''
        q="".join([icl_examples, output_template])
        return q

