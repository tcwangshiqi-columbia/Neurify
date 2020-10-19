import numpy as np
import time
from copy import deepcopy
import pickle

l2_norm = np.linalg.norm

class Embedding:
    '''
    A class representing a word embedding space.
    '''
    def __init__(self, word2index, index2word, index2embedding, blacklist_threshold = 0.9):
        '''
        Create an Embedding object.

        Parameters
        -------------
        word2index: dict
            dictionary mapping words to indexes in the embedding matrix.
        
        index2word: dict
            dictionary mapping indexes in the embedding matrix to words.
            (inverse mapping of `word2index`)

        index2embedding: dict
            dictionary mapping indexes in the embedding matrix to vectors.
        
        blacklist_threshold: float
            Ignore words with L2 norm less than `blacklist_threshold`.
            This include the <PAD> entry and several gibberish words that have word
            embeddings close to zero vector.
            
        '''
        self.word2index = word2index
        self.index2word = index2word
        self.index2embedding = index2embedding
        self.blacklist_threshold = blacklist_threshold
    

    def copy(self):
        '''
        Copy Embedding object.
        '''
        word2index_copy = deepcopy(self.word2index)
        index2word_copy = deepcopy(self.index2word)
        index2embedding_copy = deepcopy(self.index2embedding)
        return Embedding(word2index_copy, index2word_copy, index2embedding_copy)

    def replace_embeddings(self, original_embedding, new_embedding):
        '''
        Make a new Embedding object wherein the entries in `new_embedding` replace the entries
        in `original_embedding`. This is used so that counterfitted word vectors replace the 
        original word embeddings, but non-counterfitted words remain the same.

        Parameters
        -------------
        original_embedding: Embedding
            The original word embeddings.
        
        new_embedding: Embedding
            The counterfitted word vectors.


        Returns
        -------------
        updated_embedding: Embedding

        '''
        # embedding dimensions must be the same
        assert len(original_embedding.index2embedding[0]) == len(new_embedding.index2embedding[0])
        updated_embedding = original_embedding.copy()
        for word, index in new_embedding.word2index.items():
            new_word_embedding = new_embedding.index2embedding[index]
            old_index = updated_embedding.word2index[word]
            # replace old word embedding
            updated_embedding.index2embedding[old_index] = new_word_embedding
        return updated_embedding # old embedding is now updated

    def word_distance(self, word1, word2):
        ''' 
        Return the L2 distance between `word1` and `word2`.

        Parameters
        ----------------
        word1: str

        word2: str

        Returns
        ----------
        float
        '''
        index1 = self.word2index[word1]
        index2 = self.word2index[word2]
        return l2_norm(self.index2embedding[index1] - self.index2embedding[index2])

    def nearest_neighbors(self, word : str , N = 10) :
        '''
        Get the `N` nearest neighbors of `word`, and their distances.

        Parameters
        ------------
        word: str
        
        N: int
            number of nearest neighbors to compute.

        Returns
        -----------
        (nearest_words, nearest_words_distances)

        '''
        word_idx = self.word2index[word]
        word_embedding = self.index2embedding[word_idx]
        if word_idx < 3 or l2_norm(word_embedding) <= self.blacklist_threshold :
            #  word is too close to zero vector
            return [], np.array([])
        distances = np.array(len(self.index2word)*[0.0])
        for index, embedding in enumerate(self.index2embedding):
            if index < 3  or l2_norm(embedding) <= self.blacklist_threshold: 
                # ignore <PAD>, <UNK> and blacklisted words
                distance = np.inf
            else :
                distance = l2_norm(word_embedding-embedding)
            distances[index] = distance
        # get N nearest indexes, ignore word_idx
        nearest_indexes = np.argsort(distances)[1:N+1]
        nearest_words = [self.index2word[idx] for idx in nearest_indexes]
        nearest_words_distances = distances[nearest_indexes]
        return  nearest_words, nearest_words_distances

    def build_neighbors_map(self, words, N = 10, return_distances = False) :
        '''
        Get the `N` nearest neighbors of each word in `words`.
        
        Parameters
        ------------
        words: list
            List of words to compute nearest neighbors of.
        
        N: int
            Number of nearest neighbors to compute.

        return_distances: bool
            If return distances also.

        Returns
        -------------
        words_map: dict
            map {word: neighbors}
        
        distances_map: dict, optional
            map {word: distances}
            
        '''
        words = list(set(words)) # do this to deduplicate words
        words_map = dict()
        distances_map = dict()
        for word in words :
           nearest_words, distances = self.nearest_neighbors(word,N)
           words_map[word] = nearest_words
           distances_map[word] = distances
        if return_distances:
            return words_map, distances_map
        else :
            return words_map
    
    def filter_by_distance(self, words_map, distances_map, threshold = 3.0):
        '''
        Filter out neighbors with distance > `threshold`

        words_map: dict
            map {word:neighbors}

        distanced_map: dict
            map {word: distances}

        threshold: float
            the distance threshold
        '''
        filtered_words_map = deepcopy(words_map)
        for word in filtered_words_map:
            mask = np.where(distances_map[word] <= threshold)[0]
            filtered_words_map[word] = [words_map[word][i] for i in mask]
        return filtered_words_map

    def build_dist_map(self, words_map, embedding):
        '''
        Build map {word: distances}
        '''
        dist_map = dict()
        for word, neighbors in words_map.items() :
            distances = np.array([self.word_distance(word,neighbor) for neighbor in neighbors])
            dist_map[word] = distances
        return dist_map