from navec import Navec
import numpy as np

path = '../../static/embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)


def sentence_maximum(sentence_):
    sentence_ = sentence_.split(" ")
    max_embedding_ = np.array([-1.0 for _ in range(300)])
    for i in range(len(sentence_)):
        if sentence_[i] in navec:
            t1 = max_embedding_ > navec[sentence_[i]]
            t2 = max_embedding_ < navec[sentence_[i]]
            max_embedding_ = max_embedding_ * t1 + navec[sentence_[i]] * t2
    return max_embedding_


def sentence_average(sentence_):
    sentence_ = sentence_.split(" ")
    average_embedding_ = np.array([0.0 for _ in range(300)])
    for i in range(len(sentence_)):
        if sentence_[i] in navec:
            average_embedding_ += navec[sentence_[i]]
    average_embedding_ /= len(sentence_)
    return average_embedding_
