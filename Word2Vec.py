import multiprocessing

from gensim.models import Word2Vec
import main
import numpy as np
import pickle
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

f = open("data/Tokens/sard_multi_replace_tokens_no_dup.pkl", 'rb')
data_sard = np.array(pickle.load(f))
f = open("data/Tokens/git_replaced_tokens_no_dup.pkl", 'rb')
data_sard =  np.concatenate((data_sard, np.array(pickle.load(f))), axis=0)
f = open("data/Tokens/nvd_replace_tokens_no_dup.pkl", 'rb')
data_sard = np.concatenate((data_sard, np.array(pickle.load(f))), axis=0)

cores = multiprocessing.cpu_count()

w2v_model = Word2Vec(min_count=1,
                     window=3,
                     size=300,
                     workers=cores-1)


data_tokens = [line[0] for line in data_sard[:, :1]]

w2v_model.build_vocab(data_tokens, progress_per=10000)

w2v_model.train(data_tokens, total_examples=w2v_model.corpus_count, epochs=1000, report_delay=1)

out_tokens = []
for func in data_tokens:
    out_tokens.append(w2v_model[func])


with open('data/all_multi_replace_w2v.pkl', 'wb') as f:
   pickle.dump(out_tokens, f)
