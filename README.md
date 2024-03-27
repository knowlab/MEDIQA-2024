# MEDIQA-2024 Project Documentation

Welcome to the MEDIQA-2024 project repository! We share our approach for addressing the MEDIQA-CORR Shared Tasks (1, 2, and 3) leveraging the capabilities of GPT-4 combined with the Retrieval Augmented Generation (RAG) technique. Here's how we've structured our solution:

## Methodology Overview

The core of our method begins with leveraging GPT-4 to synthesize "Reasons" from the provided training dataset. We then use RAG for  train data selection process using OpenAI's embedding.

When it comes to testing and validation, we diverge slightly from the initial process. Specifically, "Reasons" are not added to the test examples, ensuring that the model's responses are generated based purely on its understanding and reasoning capabilities. This unified approach allows for simultaneous output generation across all tasks, streamlining the evaluation process.

## Data Requirements

Within the `data` directory, essential files for augmenting test and validation examples with ```Reasons```and RAG are stored for executing the tasks. To ensure the pipeline functions correctly, please populate this directory with the necessary test and validation datasets.

## Execution Instructions

### Running the Pipeline

Utilize the `run_gpt.sh` script to run the pipeline. It's important to note that access to the OpenAI API and sufficient credits are prerequisites for running this program successfully. Be sure to adjust the file paths as necessary to match your local or cloud environment.

Please provide the following arguments:

- `--train_loc` specifies the location of the training data.
- `--val_loc` defines where the test/validation data is located.
- `--model_name` allows you to choose between using GPT-4 or GPT-3.5.
- `--api_key` is where you'll input your OpenAI API key.
- `--similarity_file_loc` points to the location and name of the Similarity file, which should be placed in the data folder.
- `--reason_file_loc` indicates the location and name of the Reason file, also to be placed in the data folder.
- `--output_file_loc` designates the location and name for the output file.

