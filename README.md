## Linear Chain Conditional Random Field

#### There are 3 models:
The Original CRF: crf.py\\
The Modified CRF: crf2.py\\
The Baseline model: baseline.py\\


#### To train the CRF models:
Execute "python3 crf.py" or "python3 crf2.py" in the terminal.\\
The script will print the epoch number, log-likelihood/objective function value, and the training and development sets scores after each epoch. The script will write these same values to a *.graph file, which can be used to graph these values over the course of training. The scripts will also write the current model parameters to a *.model file after each epoch


#### To generate the test set scores:
For both CRF models, you will have to use the keyword "test" and provide the *.model file for the model that should be used for the evaluation.\\
For example: "python3 crf.py test crf_50.model" or "python3 crf2.py test crf2_50.model". \\
The script will print the test set scores to standard output.\\
For the baseline model, simply execute "python3 baseline.py" and the script will print the test set scores to standard output.\\


#### To inspect the values of the models:
For both CRF models, you will have to use the keyword "inspect", and provide the *.model file for the model you would like to inspect and the number of most significant features you would like to view from the learned weights matrices.
For example: "python3 crf.py inspect crf_50.model 20" or "python3 crf2.py inspect crf2_50.model 20". \\
The script will print the 20 most significant features from each matrix to standard output.\\

