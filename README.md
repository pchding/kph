# Key phrase extraction for scientific literature

This repo contains the end-to-end model for extracting key phrases from the records on PubMed in the mlflow folder. MLflow contain four modules that will execute the following four functions
1. Collect records from PubMed with “search term”
2. Preprocessing (generate tag sequence, extracting training dataset)
3. Train model on the training set, report validation results
4. Use trained model to extract keyphrases from records outside of training. Depending on the configuration, either store the records with extracted keyphrases in a JSON file, or push the records to the configured elasticsearch/kibana backend

Another part of the project involves building an elasticsearch/kibana backend and build a clio-lite powered contexual search engine on it. The docker-compose file for setting up the server and the code for building clio-lite based search engine. You can find related files in the es folder.

