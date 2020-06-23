# End-to-end Key Phrase Extraction Model powered by MLflow

There are three modules in this model:
1. pull_pmed_data: it pulls all the records related to the defined search term from PubMed and save them in a JSON file
2. prep_data: it cleans the records and genertate the tag seauence for future training.
3. ke_train: it uses the cleanned sequence with key phrases to train a bi-LSTM-CRF model and then use this model to augment the records without key phrases with extracted key phrases. It could also write these records into an ELasticsearch database.

## Running the model

Make sure you have installed MLflow package in your python environment. Download all the files, then go to the mlflow folder.

To run each module, exectute the following command to pull pubmed data using parameter defined in the input file (input.txt)
```
mlflow run . -e pull_pmed_data -P inputfile=1
```
Substitute 'pull_pmed_data' to any other modules, if you do not want use the inputfile and want set paramters through command line, you could execute something the following command
```
mlflow run . -e pull_pmed_data -P inputfile=0 -P search_term='covid' -P max_records=100000
```

To run the whole end-to-end model

```
mlflow run . -e main -P inputfile=1
```

Sample input.txt is also included.
