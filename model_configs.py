config_sglsen_1 = {'hidden_size':300,
                   'num_lstm_layers':2,
                   'hidden_dropout_prob':0.3,
                   'MLP_hidden':512,
                   }


class SimpleConfig(object):
    def __init__(self, vocab, config_dict):
        self.vocab_size = len(vocab)
        self.hidden_size = config_dict['hidden_size']
        self.num_lstm_layers = config_dict['num_lstm_layers']
        self.hidden_dropout_prob = config_dict['hidden_dropout_prob']
        self.MLP_hidden = config_dict['MLP_hidden']

