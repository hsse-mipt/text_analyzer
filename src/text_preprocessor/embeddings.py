from navec import Navec
import numpy as np

path = '../../static/Embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)


def maximum_embedding(sentence_):
    max_embedding_ = np.array([-1.0 for _ in range(300)])
    for i in range(len(sentence_)):
        for j in range(300):
            max_embedding_[j] = max(max_embedding_[j], navec[sentence_[i]][j])
    return max_embedding_


def average_embedding(sentence_):
    average_embedding_ = np.array([0.0 for _ in range(300)])
    for i in range(len(sentence_)):
        average_embedding_ += navec[sentence_[i]]
    average_embedding_ /= len(sentence_)
    return average_embedding_
