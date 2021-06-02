from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from Vectorizer import Vectorizer

import telebot, joblib, string

import wget, zipfile, gensim

import numpy as np

from functools import lru_cache
from pymystem3 import Mystem


# snowball = SnowballStemmer(language="russian")
# russian_stop_words = stopwords.words("russian")

vectorizer = Vectorizer(model_file = "204.zip")

def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens

TOKEN = ''

# model_ins = joblib.load("data/models/model_ins_logistic_2.joblib")
# model_thr = joblib.load("data/models/model_thr_logistic_2.joblib")
# model_obs = joblib.load("data/models/model_obs_logistic_2.joblib")

model_ins = joblib.load("data/models/model_ins_w2v_CBC.joblib")
model_thr = joblib.load("data/models/model_thr_w2v_CBC.joblib")
model_obs = joblib.load("data/models/model_obs_w2v_CBC.joblib")

лbot = telebot.TeleBot(TOKEN)

print("Bot has been started.")

@bot.message_handler(content_types = ['text'])
def reply(message):
	text = message.text

	text_vec = vectorizer.Vectorize_one(text)

	result_ins = round(model_ins.predict_proba([text_vec]).T[1][0], 4)
	result_thr = round(model_thr.predict_proba([text_vec]).T[1][0], 4)
	result_obs = round(model_obs.predict_proba([text_vec]).T[1][0], 4)
	
	print("Text: {}".format(text))
	print(f"INSULT: {result_ins}; THREAT: {result_thr}; OBSCENITY: {result_obs}")
	print('------')

	if (result_ins < 0.5 and result_thr < 0.5 and result_obs < 0.5): return
	# if (result_ins < 0.5): return

	bot.reply_to(message, f"Это оскорбление с вероятностью {result_ins}\nЭто угроза с вероятностью {result_thr}\nЭто домогательство с вероятностью {result_obs}\n")
	# bot.reply_to(message, f"Это оскорбление с вероятностью {result_ins}")
	

bot.polling()