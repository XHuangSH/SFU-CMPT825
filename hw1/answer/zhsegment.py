import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        candidates = ([first]+self.segment(rem) for first,rem in self.splits(text))
        return max(candidates, key=self.Pwords)

    def splits(self, text, L=3):
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[:i+1], text[i+1:])
                for i in range(min(len(text), L))]

    def Pwords(self, words):
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)


# punish long word 
def punishLong(word, N):
    "Estimate the prob of unknown word while accounting for word length"
    return 10. / (N * 8000 ** (len(word)))


# entry class defination 
class Entry(object):
    def __init__(self, word, startPos, logProb, backPoint):
        super(Entry, self).__init__()
        self.word = word
        self.startPos = startPos
        self.logProb = logProb
        self.backPoint = backPoint

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return (self.startPos == other.startPos) and (self.word == other.word) and (self.logProb == other.logProb)

    def __lt__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        # if self.startPos == other.startPos:
        return self.logProb > other.logProb
        # else:
        #    return self.startPos < other.startPos

    def __gt__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.logProb <= other.logProb
        # return self.startPos > other.startPos

    def print(self):

        print("Word: {} Start: {} End: {} Log Prob: {} Back: {}".format(self.word, self.startPos,
                                                                        self.startPos + len(self.word), self.logProb,
                                                                        self.backPoint))



def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name, encoding='utf-8') as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)



######################################################
## implementation of bigram with unigram smoothing ###
######################################################
def segChinese(refLine, segmenter_uni, segmenter_bi, L):
    # Gather all possible inputs at position 0
    chart = {}
    heap_q = []
    threshold = 3
    #########################################################
    ## initialize the heapq by adding all words at index 0 ##
    #########################################################
    # push words into heapq
    for s in segmenter_uni.splits(refLine, L):
        if s[0] in segmenter_uni.Pw or len(s[0]) <= threshold:  # use uni-gram segmenter for splitting
            heapq.heappush(heap_q, Entry(s[0], 0, math.log(segmenter_uni.Pw(s[0])), None))

    # initialize chart
    for i in range(len(refLine)):
        chart[i] = Entry(None, None, None, None)

    endIdx = 0

    ##################################################################
    ## interative loop, find best entry from heapy and update chart ##
    ##################################################################
    while len(heap_q) > 0:
        # pop best entry (might reconsider older entry)
        entry = heapq.heappop(heap_q)
        endIdx = entry.startPos + len(entry.word) - 1

        # update chart to current best entry
        if chart[endIdx].backPoint is not None:
            prevEntry = chart[endIdx].backPoint
            if entry.logProb > prevEntry.logProb:
                chart[endIdx] = entry
            if entry.logProb <= prevEntry.logProb:
                continue
        else:
            chart[endIdx] = entry


        #####################################################
        ## update log prob according to the new best entry ##
        #####################################################
        h = []
        for s in segmenter_uni.splits(refLine[endIdx +1:], L):

            if s[0] in segmenter_uni.Pw or len(s[0]) <= threshold:
                bi_gram_word = entry.word +" " +s[0]
                uni_gram_prob = segmenter_uni.Pw(s[0])
                
                if bi_gram_word in segmenter_bi.Pw: # use bi-gram if possible
                    assert entry.word in segmenter_uni.Pw # assert first word appears in 1w dictionary
                    bi_gram_prob = (segmenter_bi.Pw(bi_gram_word)*Pw_bi.N) / (segmenter_uni.Pw(entry.word)*Pw_uni.N)
                    h.append(Entry(s[0], endIdx + 1, entry.logProb + math.log(0.1*bi_gram_prob+0.9*uni_gram_prob), entry)) 
                    
                else: # else bi-gram has probability zero, smoothing with unigram
                    h.append(Entry(s[0], endIdx + 1, entry.logProb + math.log(0.1*0.0+0.9*uni_gram_prob), entry)) 

        for s in h: # only add different entry
            if s not in heap_q:
                heapq.heappush(heap_q, s)

    cWord = chart[len(chart) - 1]
    cWord
    res = []
    while cWord.backPoint != None:
        res.append(cWord.word)
        cWord = cWord.backPoint
    res.append(cWord.word)
    setSeg = res[::-1]
    return setSeg







if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    L = 6
    Pw_uni = Pdist(data=datafile(opts.counts1w), missingfn=punishLong)
    Pw_bi = Pdist(data=datafile(opts.counts2w), missingfn=punishLong)
    segmenter_uni = Segment(Pw_uni)
    segmenter_bi = Segment(Pw_bi)

    with open(opts.input) as f:
        for line in f:
            print(" ".join(segChinese(line.strip(), segmenter_uni, segmenter_bi, L)))
