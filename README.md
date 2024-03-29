# Documentation on ```KnowLab``` Submission for MEDIQA 2024 

Welcome to the KnowLab submission repository for MEDIQA-2024. We share our approach for addressing the MEDIQA-CORR Shared Tasks (1, 2, and 3) leveraging the capabilities of GPT-4 combined with the Retrieval Augmented Generation (RAG) technique. Here's how we've structured our solution:

## Methodology Overview
We have develeped two systems which we term the first system and the second system. The first system is in: 
https://github.com/wuzl01/Knowlab_MEDIQA-CORR-2024

Our final submission is a rule based ensemble by using the two systems. To run the ensemble system please use the ```ensemble-u.sh``` script. It's decription can be found at the end of this page. 

In this repository we have the second system which has shown to be effective in the generation task. We describe the second system briefly here. 

The core of our method begins with leveraging GPT-4 to synthesize "Reasons" from the provided training dataset. We then use RAG for  train data selection process using OpenAI's embedding. 

When it comes to testing and validation, we diverge slightly from the initial process. Specifically, "Reasons" are not added to the test examples, ensuring that the model's responses are generated based purely on its understanding and reasoning capabilities. This unified approach allows for simultaneous output generation across all tasks, streamlining the evaluation process. Our system ensembled with simple prompting technique improves on the NLG scores.

## Data Requirements

Within the `data` directory, essential files for augmenting test and validation examples with ```Reasons```and RAG are stored for executing the tasks. To ensure the pipeline functions correctly, please populate this directory with the necessary test and validation datasets.

## Execution Instructions

### Running the Pipeline

Utilize the `gpt-u.sh` script to run the pipeline. It's important to note that access to the OpenAI API and sufficient credits are prerequisites for running this program successfully. Be sure to adjust the file paths as necessary to match your local or cloud environment.

Please provide the following arguments:

- `--train_loc` specifies the location of the training data.
- `--val_loc` defines where the test/validation data is located.
- `--model_name` allows you to choose between using GPT-4 or GPT-3.5.
- `--api_key` is where you'll input your OpenAI API key.
- `--similarity_file_loc` points to the location and name of the Similarity file, which should be placed in the data folder.
- `--reason_file_loc` indicates the location and name of the Reason file, also to be placed in the data folder.
- `--output_file_loc` designates the location and name for the output file.

### Running the ensemble 

Utilize the `ensemble-u.sh` script to run the ensemble pipeline where we combine results from two systems using a rule-based approach.

Please provide the following arguments:
- `--train_loc` specifies the location of the training data.
- `--val_loc` defines where the test/validation data is located.
- `--model_name` allows you to choose between using GPT-4 or GPT-3.5.
- `--api_key` is where you'll input your OpenAI API key.
- `--similarity_file_loc` points to the location and name of the Similarity file, which should be placed in the data folder.
- `--reason_file_loc` indicates the location and name of the Reason file, also to be placed in the data folder.
- `--system1_loc`  Run file from system 1 that handles Task 1 and Task 2.
- `--system2_loc` Run file from the system 2 that handles Text generation. 
- `--pass_number` 0 for ensembling. We do not sample examples for ensembling. 
- `--output_file_loc` designates the location and name for the output file.