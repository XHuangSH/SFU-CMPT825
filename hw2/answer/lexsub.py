import os, sys, optparse
import tqdm
import pymagnitude
import numpy as np
import glob
## lexFiles: can be a list of files
## return: neighboring words for each word 
## add all pairwise related neighbors
def get_neighbor_ws(lexFiles):
    word2neigh = {}
    accountedFor = set()
    for lexDex, lexF in enumerate(lexFiles):
        #print("On [{}/{}]".format(lexDex, len(lexFiles)))
        with open(lexF, 'r') as f:
            for line in f:
                cWords = line.lower().strip().split(' ')
                for i in range(len(cWords)): 
                    w1 = cWords[i]  # current word
                    for j in range(i+1, len(cWords)):
                        w2 = cWords[j]  # later word
                        if w1 not in word2neigh:  
                            word2neigh[w1] = [w2]
                        else:
                            if w2 not in word2neigh[w1]:
                                word2neigh[w1].append(w2)
                        if w2 not in word2neigh:  
                            word2neigh[w2] = [w1]
                        else:
                            if w1 not in word2neigh[w2]:
                                word2neigh[w2].append(w1)
    return word2neigh


## lexFiles: can be a list of files
## return: neighboring words for each word 
## add all non-pairwise related neighbors
def get_oneside_neighbor_ws(lexFiles):
    word2neigh = {}
    accountedFor = set()
    #lexFiles = [lexFiles[-2]]
    #lexFiles = [lexFiles[-2], lexFiles[1]]
    for lexDex, lexF in enumerate(lexFiles):
        #print("On [{}/{}]".format(lexDex, len(lexFiles)))
        with open(lexF, 'r') as f:
            for line in f:
                cWords = line.lower().strip().split(' ')
                tmpL = [word for word in cWords[1:]]
                if cWords[0] not in word2neigh:
                    word2neigh[cWords[0]] = tmpL
                else:
                    word2neigh[cWords[0]] = word2neigh[cWords[0]] + tmpL

    return word2neigh


## wVecs: word vectors
## word2Neigh: adjacent words according to lexicons
## beta, alpha: hyper-parameters
## numIters: number of iterations
def retrofit(wVecs, word2Neigh, beta = 1.0, alpha = 1.0, numIters = 10):
    
    givenVocab = set()
    for word, vect in wVecs:
        givenVocab.add(word)

    # Need to create a modifiable version of the vectors as pymag is read only??!?!?!?
    newEmbs = {}
    for word, emb in wVecs:
        newEmbs[word] = emb
        
    for itCount in range(numIters):
        for wCount, gWord in enumerate(givenVocab):
            #if (wCount % 100000) == 0:
                #print("Iter [{}/{}] Word [{}/{}]".format(itCount, numIters, wCount, len(givenVocab)))
          
            tmpEmb = np.zeros(newEmbs[gWord].shape)
            if gWord in word2Neigh:
                nCount = 0
                cLoopVocab = word2Neigh[gWord]
                for word in cLoopVocab:
                    if word in newEmbs:
                        tmpEmb += beta * newEmbs[word]
                        nCount += 1
                
                newEmbs[gWord] = ((tmpEmb + (alpha * wVecs.query(gWord)))) / (nCount + alpha)
    
    return newEmbs

class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn



    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    senseMag = LexSub(os.path.join('data/','glove.6B.100d.magnitude'))
    senseVecs = senseMag.wvecs
    lexFiles = glob.glob('data/lexicons/ppdb-xl.txt')
    word2neigh_oneside = get_oneside_neighbor_ws(lexFiles)
    newWordVects_oneside = retrofit(senseVecs, word2neigh_oneside, beta=1.0, alpha=2.0)
    np.save("newWordVects_oneside_wn.npy", newWordVects_oneside)
    with open("newVects_oneside_wn.txt", "w") as f:
        count = 0
        vSize = len(newWordVects_oneside)
        for word, emb in newWordVects_oneside.items():
            #if (count % 100000) == 0:
                #print("Writing to Txt [{}/{}]".format(count, vSize))
            
            f.write(word)
            for num in emb:
                f.write(" " + str(num))
            f.write("\n")
            count += 1

    # create pymagnitude file
    import subprocess
    subprocess.call(["python", "-m", "pymagnitude.converter", "-i", "newVects_oneside_wn.txt", "-o", "newVects_oneside_wn.magnitude"])


    lexsub = LexSub('newVects_oneside_wn.magnitude')
    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
