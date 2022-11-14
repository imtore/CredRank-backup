from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from pathlib import Path

word2vec_output_file = 'glove.word2vec'

if not Path("glove.word2vec").exists():
    glove_input_file = '../../../../datasets/glove/glove.twitter.27B.50d.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)
    print('glove transformed')


# load the Stanford GloVe model
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print('glove loaded')




