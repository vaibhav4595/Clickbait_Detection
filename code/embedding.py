from gensim.models import Doc2Vec, doc2vec
from ast import literal_eval
import pdb
from tqdm import tqdm
import json
import pickle

class Embedding():

    def __init__(self, sents, articleId):
        self.sents = sents
        self.articleId = articleId
        self.labelledSents = []

    def label(self):
        for i in range(len(self.sents) / 3):
            self.labelledSents.append(doc2vec.LabeledSentence(words=self.sents[3*i].split(), tags=['postText_%s' % self.articleId[i]]))
            self.labelledSents.append(doc2vec.LabeledSentence(words=self.sents[3*i+1].split(), tags=['targetTitle_%s' % self.articleId[i]]))
            self.labelledSents.append(doc2vec.LabeledSentence(words=self.sents[3*i+2].split(), tags=['targetDescription_%s' % self.articleId[i]]))

    def train(self):
        self.model = Doc2Vec()
        self.model.build_vocab(self.labelledSents)
        for i in tqdm(range(10)):
            self.model.train(self.labelledSents)


if __name__ == '__main__':

    fp = open('../../clickbait17-validation-170630/instances.jsonl')
    all_lines = fp.readlines()
    lines = all_lines[:int(len(all_lines)*0.7)]
    sents = []
    articleId = []
    for line in lines:
        d = literal_eval(line)
        articleId.append(d['id'])
        sents.append(d['postText'][0])
        sents.append(d['targetTitle'])
        sents.append(d['targetDescription'])

    article_embed = {}
    
    e = Embedding(sents, articleId)
    e.label()
    e.train()
    e.model.save('../data/embed_model')
    
    for k in tqdm(articleId):
        article_embed['postText_%s' % k] = e.model.docvecs['postText_%s' % k]
        article_embed['targetTitle_%s' % k] = e.model.docvecs['targetTitle_%s' % k]
        article_embed['targetDescription_%s' % k] = e.model.docvecs['targetDescription_%s' % k]
   
    fp = open('../data/article_embed.pkl', 'w')
    pickle.dump(article_embed, fp)
