# NFT-Recommender-MGAT

### config
- will be updated

### dataset
- **collections (MAIN)**
  - For each NFT collections, there are interactions and features files included.
  - For example, train.npy is user-item interactions and image_feat.npy is preprocessed image features of items.
- user_features
  - Raw data of user features.
  - For each user, number of transactions, average transaction price, and average holding period are included.
- item_features
  - Raw data of item features.
  - For each item, there are image, text, price, and transaction features included.

### runfile
- Contains shell scripts that can be used to run the main file.

### saved
- The best model that shows the lowest valid metric during the model training process is saved.

### Create_dataset.ipynb
- Code to create the input data file in the format required by our model.

### DataLoad.py
- Creates a dataloader of interaction data.
- You can adjust the number of negative samples during training by the parameter "pop_num".

### GraphGAT.py
- An attention-based graph convolutional networks which is used in Model.py
- You can adjust the dropout ratio during training by the parameter "DROPOUT"

### Model.py (MAIN)
- A multi-modal graph-based, multi-objective model for recommending NFTs. (named "MMMO")

### main.py (MAIN)
- The code to run experiments using the model.
