# reuters-articles-machine-learning
Perform different machine learning techniques on dataset of reuters articles (using Python). 

This repository contains Python scripts that use machine learning techniques on a dataset of 20,000 Reuters articles.

Repository Contents:

+sgm-files: Folder that contains the source files for the articles in sgm format.

+minhash.py: Contains a minhash implementation on a feature vector that was generated from the sgm files. First the script calculates the true Jaccard Similarity between each vector. Then it calculates the minhash similarity. The times for the two implementations are then printed to the screen.

+pickled_minhash: folder that contains pickled version of the feature vector that is used in the minhash algorithm.
