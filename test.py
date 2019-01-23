from model.networks import *
Model = C_BLSTM([2, 2, 25])
Model.test(data_path='.\\data\\assignment_training_data_word_segment.json')
