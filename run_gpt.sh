python run_mediqa_gen.py \
 --train_loc data/MEDIQA-CORR-2024-MS-TrainingData.csv \
 --val_loc data/March_26_2024_Official_Test_Set_MEDIQA-CORR.csv \
 --model_name gpt-4 \
 --api_key api-key \
 --similarity_file_loc data/Offical_Test_set_OpenAI_similariy.json \
 --reason_file_loc data/MSTrain_Reasons_from_chat_gpt_4.json \
 --output_file_loc data/test_github.json \