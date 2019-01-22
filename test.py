from model.networks import *
Model = C_BLSTM([2, 1, 25])
Model.test(data_path='.\\data\\assignment_training_data_word_segment.json')
