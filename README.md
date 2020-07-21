# Key phrase extraction for scientific literature

This repo contains the end-to-end model for extracting key phrases from the records on PubMed in the mlflow folder. MLflow contain four modules that will execute the following four functions
1. Collect records from PubMed with “search term”
2. Preprocessing (generate tag sequence, extracting training dataset)
3. Train model on the training set, report validation results
4. Use trained model to extract keyphrases from records outside of training. Depending on the configuration, either store the records with extracted keyphrases in a JSON file, or push the records to the configured elasticsearch/kibana backend

Please be noted, the PubMed API wrapper has several unfixed bugs, MLflow has been configured to install an unofficial version published at https://github.com/iacopy/pymed/tree/fork-fixes. 

Another part of the project involves building an elasticsearch/kibana backend and build a clio-lite powered contexual search engine on it. The docker-compose file for setting up the server and the code for building clio-lite based search engine (the official version of clio-lite does not support Elasticsearch version 7+ at the time, so an unofficial version at https://github.com/inactivist/clio-lite/tree/bugfix/fix-typeerror-basic-usage is used). You can find related files in the es folder.

Notebooks used during the experimentation phase are included in the notebooks folder. You can find both LSTM and LSTM-crf models there. A different crf model that includes a context windows to explicitly consider the effects from nearby words is also in that folder, although this model does not perform better than the LSTM-CRF model (since LSTM already encompass this information).

