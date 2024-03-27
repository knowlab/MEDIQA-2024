import re
import json
from collections import Counter
import pandas as pd
import argparse
def find_majority_element(lst):
    # Count the occurrences of each element
    if not lst:  # Check if the list is empty
        return 0
    counts = Counter(lst)
    # Find the element(s) with the highest count
    max_count = max(counts.values())
    majority_elements = [key for key, value in counts.items() if value == max_count]
    
    # Check if there is a tie
    if len(majority_elements) > 1:
        return 0  # Return 0 if there is a tie
    else:
        return majority_elements[0]  # 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_loc", type=str, required=True,
                        help="GPT4 output file")
    parser.add_argument("--run_file_loc", type=str, required=True,
                        help="Run submission file location")
    args = parser.parse_args()
    output_file_loc=args.output_file_loc
    run_file_loc=args.run_file_loc
    pattern1 = r"Error Flag: (\d+)"
    pattern2=r"Error Sentence ID: (-?\d+)"
    pattern3=r"Corrected Sentence: (.+)"
    with open(output_file_loc, "r") as f:
        results=json.load(f)
    prediction_results=[]
    for text_id in results:
        rs_set=results[text_id]
        r1=[]
        r2=[]
        r3=[]
        for rs in rs_set:
            text = rs_set[rs]
            # Find all occurrences of the pattern in the text
            print(text)
            matches1 = re.findall(pattern1, text)
            matches2=re.findall(pattern2, text)
            matches3=re.findall(pattern3, text)
            try:
                r1.append(matches1[0])
                r2.append(matches2[0])
                r3.append(matches3[0])

            except:
                continue
        #print(text_id, r1)
        #print(text_id, r2)
        #print("Majority vote: ", find_majority_element(r1))
        #print("Majority vote: ", find_majority_element(r2))
        try:
            corrected_sent=r3[0]
        except:
            corrected_sent="NA"
        result={"id":text_id,
                "error_flag":int(find_majority_element(r1)), 
                "error_sent":int(find_majority_element(r2)),
                "correct_sent":corrected_sent
                }
        prediction_results.append(result)
    with open(run_file_loc, "w") as f:
        for elem in prediction_results:
            elem_list=[elem[k] for k in elem]
            line = ' '.join(map(str, elem_list))    
            f.write(f"{line}\n")
