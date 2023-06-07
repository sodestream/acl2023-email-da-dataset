# acl2023-email-da-dataset
Data and code for the paper "LEDA: a Large-Organization Email-Based Decision-Dialogue-Act Analysis Dataset" at ACL 2023

Instructions:

1. We recommend creating a fresh python (3.8.0) environment to run this code. The exact versions of the libraries used can be found in "requirements.txt".
2. The data and python notebook to generate the plots and some of the statistics is in the "Notebook" folder, just open it and execute the chunks in order.
   - the Notebook will also generate an input file for the dialog tagger, but you can find the same file already precomputed in the "PredictionModel/data" folder
3. The code to run the prediction models is in the "PredictionModels/model" folder. From there, execute "./run.sh" and the dev/test set scores for each epoch should be generated in log.txt.
   - we recommend running this on GPUs with at least 16GB of RAM (in case of memory errors, reducing the batch size could help)

In case of issues please contact m.karan@qmul.ac.uk.
