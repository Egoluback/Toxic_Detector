# Toxic_Detector
ML-bot that detects toxicity in russian texts.

Works on TeleBot(telegram API) <br />
Data: [Train&Test](https://www.kaggle.com/alexandersemiletov/toxic-russian-comments) [Test](https://www.kaggle.com/blackmoon/russian-language-toxic-comments) <br />
Bot represents 3 models classifying insults, threats and obscenities.

# How it works?
All words in sentences are presented in vectors with word2vec model([Model 204 on nlppl.eu, trained on RNC, Wikipedia, News and ARM](http://vectors.nlpl.eu/repository/)). The resulting vector of the proposal is the average value of its vectors. <br />
Train data shape: Nx300. I use 3 CatBoostClassifier models to train on insults, threats and obscenities datasets.
# Result
This architecture is nice for getting main topic of sentence(because mean word2vec vector guesses semantics well), but it is not perfect for predicting tone of sentence. For this task it's better to use different way to vectorize sentences and different models(not decisions trees, better NN(RNN or CNN)). <br />
Maybe I'll come back to this task later with better method.
