
The scripts uses relative paths therefore please run them from the directory that contains the codes.
The data should be placed under ./data/ Scripts will check that directory for data.

INSTALLATION:
=============
Required non default packages for python:
numpy
sklearn
matplotlib

Also it is possible to create a conda environment with the environment.yml usign the following commands:

conda env create -f environment.yml
conda activate assignment

Using the conda environment also makes you able to use the same version with the author. 
If you are don't have conda you may want to install the compact version from the following link:
https://docs.conda.io/en/latest/miniconda.html

USAGE:
======
"network.py" has the required functions and network class inside

"python train.py" command start the training and for the given hyperparameters in the script. For each epoch it saves the model to ./models/bs_<batch_size>_lr_<learninig_rate>/ It also keep record of the best_model as BestModel.pkl in the this folder. When the epochs are calculated it also saves the accuracy and loss plots to the same directory. Additioaly it keeps the loss and accuracy for each epoch in ./models/summary_bs_<batch_size>_lr_<learninig_rate>.pkl. Finally when the all models are trained it dumps a file called ./models/best_models.pkl where the best validation accuracy for each hyper parameter pair is stored in a dictionary

"python eval.py" command reads the ./model.pkl and loads the model it evaluates it in train, validation and test data and reports the accuracy and loss values. It also gives prediction to examples.

"python tsne.py" command again reads ./model.pkl forms the word embeddings, apply tsne and saves the 2d plot to ./tsne_plot.png

