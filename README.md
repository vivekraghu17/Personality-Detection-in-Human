# Deep Learning-Based Document Modeling for Personality Detection from Text

This code detects 5 traits (Big Traits)

-   Extroversion
-   Neuroticism
-   Agreeableness
-   Conscientiousness
-   Openness


## Requirements

-   Python 2.7
-   Theano 0.7 (Tested)
-   Pandas 18.0 (Tested)
-   Pre-trained [GoogleNews word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) vector


## Preprocessing

`process_data.py` prepares the data for training. It requires three command-line arguments:

1.  Path to [google word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) file (`GoogleNews-vectors-negative300.bin`)
2.  Path to `essays.csv` file containing the annotated dataset
3.  Path to `mairesse.csv` containing [Mairesse features](http://farm2.user.srcf.net/research/personality/recognizer.html) for each sample/essay

This code generates a pickle file `essays_mairesse.p`.

Example:

```sh
python process_data.py ./GoogleNews-vectors-negative300.bin ./essays.csv ./mairesse.csv
```


## Training

`conv_net_train.py` trains and tests the model. It requires three command-line arguments:

1.  **Mode:**
    -   `-static`: word embeddings will remain fixed
    -   `-nonstatic`: word embeddings will be trained
2.  **Word Embedding Type:**
    -   `-rand`: randomized word embedding (dimension is 300 by default; is hardcoded; can be changed by modifying default value of `k` in line 111 of `process_data.py`)
    -   `-word2vec`: 300 dimensional google pre-trained word embeddings
3.  **Personality Trait:**
    -   `0`: Extroversion
    -   `1`: Neuroticism
    -   `2`: Agreeableness
    -   `3`: Conscientiousness
    -   `4`: Openness

Example:

```sh
python conv_layer_train.py -static -word2vec 2
```
