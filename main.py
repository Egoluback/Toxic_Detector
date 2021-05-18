from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

import telebot, joblib, string

snowball = SnowballStemmer(language="russian")
russian_stop_words = stopwords.words("russian")

def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens

TOKEN = '1881824091:AAH6B7eYZnkB2WfupGUlt9ZUyre1-ixnQn0'

model_ins = joblib.load("data/model_ins_logistic_2.joblib")
model_thr = joblib.load("data/model_thr_logistic_2.joblib")
model_obs = joblib.load("data/model_obs_logistic_2.joblib")

bot = telebot.TeleBot(TOKEN)

print("Bot has been started.")

@bot.message_handler(content_types = ['text'])
def reply(message):
	text = message.text
	# result_ins = model_ins.predict([text])[0]
	# result_thr = model_thr.predict([text])[0]
	# result_obs = model_obs.predict([text])[0]

	result_ins = round(model_ins.predict_proba([text]).T[1][0], 4)
	result_thr = round(model_thr.predict_proba([text]).T[1][0], 4)
	result_obs = round(model_obs.predict_proba([text]).T[1][0], 4)
	
	print("Text: {}".format(text))
	print("INSULT: {}; THREAT: {}; OBSCENITY: {}".format(result_ins, result_thr, result_obs))
	print('------')

	if (result_ins < 0.5 and result_thr < 0.5 and result_obs < 0.5): return
	
	ans = f"Это оскорбление с вероятностью {result_ins}\nЭто угроза с вероятностью {result_thr}\nЭто домогательство с вероятностью {result_obs}\n"

	bot.reply_to(message, ans)
	

bot.polling()