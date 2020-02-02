import numpy as np
# import sys,os
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from feed_data import RelationEntityBatcher
from grapher import RelationEntityGrapher
#
# from ..data.feed_data import RelationEntityBatcher
# from ..data.grapher import RelationEntityGrapher
from environment import Episode, Env


from options import read_options
import json

#TEST MAIN
options = read_options()

data_input_dir="datasets/data_preprocessed/countries_S3/"
vocab_dir="datasets/data_preprocessed/countries_S3/vocab"
total_iterations=1000
path_length=3
hidden_size=2
embedding_size=2
batch_size=128
beta=0.1
Lambda=0.02
use_entity_embeddings=1
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/countries_s3/"
model_load_dir="nothing"
load_model=0
nell_evaluation=0

options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))

mid_to_word = {}
print('Total number of entities {}'.format(len(options['entity_vocab'])))
print('Total number of relations {}'.format(len(options['relation_vocab'])))
save_path = ''

params = options
train_environment = Env(params, 'train')
#dev_test_environment = Env(params, 'dev')
#test_test_environment = Env(params, 'test')

for episode in train_environment.get_episodes():
    # get initial state
    state = episode.get_state()
    