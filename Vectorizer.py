import wget, zipfile, gensim

import numpy as np
import pandas as pd

from functools import lru_cache
from pymystem3 import Mystem

class Vectorizer:
  def __init__(self, model_file: str = ""):
    self.mapping = {'A': 'ADJ', 'ADV': 'ADV', 'ADVPRO': 'ADV', 'ANUM': 'ADJ', 'APRO': 'DET', 'COM': 'ADJ', 'CONJ': 'SCONJ', 'INTJ': 'INTJ', 'NONLEX': 'X', 'NUM': 'NUM', 'PART': 'PART', 'PR': 'ADP', 'S': 'NOUN', 'SPRO': 'PRON', 'UNKN': 'X', 'V': 'VERB'}

    self.mystem = Mystem()

    self.model_vv = None

    if (len(model_file) == 0): self.download_model()
    else: self.load_model(model_file)
  
  def download_model(self):
    # udpipe url: https://rusvectores.org/static/models/udpipe_syntagrus.model
    model_url = 'http://vectors.nlpl.eu/repository/20/204.zip'

    m_ = wget.download(model_url)

    model_file = model_url.split('/')[-1]

    print(f"Model downloaded from {model_url}\n Path: {model_file}")

    self.load_model(model_file)

  def load_model(self, model_file: str):
    with zipfile.ZipFile(model_file, 'r') as archive:
      stream = archive.open('model.bin')
      self.model_vv = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

    print("Model loaded.")

  @lru_cache(maxsize=None)
  def tag_mystem(self, text: str) -> list:
    try:
      processed = self.mystem.analyze(text)
    except:
      return []
    tagged = []

    for w in processed:
      try:
        lemma = w["analysis"][0]["lex"].lower().strip()
        pos = w["analysis"][0]["gr"].split(',')[0]
        pos = pos.split('=')[0].strip()
        if pos in self.mapping:
            tagged.append(lemma + '_' + self.mapping[pos])
        else:
            tagged.append(lemma + '_X')
      except KeyError:
        continue
      except IndexError:
        continue
    return tagged
  
  def tokenize_word2vec(self, sentence: str) -> list:
    tokens = self.tag_mystem(sentence)
    vectors = []

    for token in tokens:
      try:
        vectors.append(self.model_vv[token])
      except:
        continue

    if (len(vectors) == 0): return self.model_vv["слэнг_NOUN"]

    return np.mean(vectors, axis = 0)
  
  def Vectorize_one(self, text: str) -> np.array:
    self.mystem = Mystem()
    return self.tokenize_word2vec(text)

  def Vectorize_corpus(self, dataset: pd.DataFrame) -> pd.DataFrame:
    self.mystem = Mystem()
    
    vectorized = []

    for object_index in range(len(dataset)):
      if (object_index % 10000 == 0): print(f"Current progress: {object_index / len(dataset) * 100}%")
      
      vector = self.tokenize_word2vec(dataset.iloc[object_index])

      vectorized.append(vector)
    
    return pd.DataFrame(list(vectorized))