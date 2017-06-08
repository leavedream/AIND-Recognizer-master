import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    # for each test word
    for word_id in range(0,len(test_set.get_all_Xlengths())):
        current_X, current_length = test_set.get_item_Xlengths(word_id)
        
        best_score = float("-inf")   # the highest score
        best_word = None             # the best guess word
        
        prob_dic = {}
        for word, model in models.items():
            try:
                logL = model.score(current_X, current_length)
                
                prob_dic[word] = logL
                if best_score < logL:
                    best_score = logL
                    best_word = word
            except:
                prob_dic[word] = float("-inf")
                
        
        probabilities.append(prob_dic)
        guesses.append(best_word)
    
    return probabilities, guesses
    
    raise NotImplementedError
