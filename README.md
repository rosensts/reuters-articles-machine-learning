# Reuters-Articles-Machine-Learning
### Perform different machine learning techniques on dataset of reuters articles (using Python). 

Author: Samuel Rosenstein

This repository contains Python scripts that use machine learning techniques on a dataset of 20,000 Reuters articles.

Required Python libraries:
    - NLTK
    - BeautifulSoup
    - Scikit-learn

After cloning the repository, simply run the scripts from the command line using Python3. (Ex. '$> python3 generate_feature_vectors.py')

The original feature matricies were created using the following steps: (The implementation can be found in the 'generate_feature_vectors.py' file.)
a. Created FreqDist of all words in corpus
b. Eliminated Stop words
c. Stemmed words
d. Only kept words that appeared more than 4 times are were longer than 4 characters
e. ***For small data set only*** took the top 200 most common words from (d)

Repository Contents:

+ 'sgm-files': Folder that contains the source files for the articles in sgm format.

+ 'generate_feature_vectors.py': generates the numeric feature vectors from the Reuters dataset. The script prints the number of feature words as the preprocessing takes place.

+ 'minhash.py': Contains a minhash implementation on a feature vector that was generated from the sgm files. First the script calculates the true Jaccard Similarity between each vector. Then it calculates the minhash similarity. The times for the two implementations are then printed to the screen.
Note: I did not implement the MinHash class. See code comment for source.

+ 'pickled_minhash': folder that contains pickled version of the feature vector that is used in the minhash algorithm.

Note: The dataset is contained within the repository so it will take a bit longer to clone.