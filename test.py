from model.networks import *
Model = BLSTM([2, 1, 5])
Model.test(data_path='.\\data\\assignment_training_data_word_segment.json')
