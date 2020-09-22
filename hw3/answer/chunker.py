# Code adapted from original code by Robert Guthrie

import os, sys, optparse, gzip, re, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import numpy as np

def read_conll(handle, input_idx=0, label_idx=2):
    conll_data = []
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
    contents = contents.rstrip()
    for sent_string in contents.split('\n\n'):
        annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
        assert(input_idx < len(annotations))
        if label_idx < 0:
            
            charTups = []
            for word in annotations[input_idx]:
        
                first = mid = last = None
                refLen = len(word)
                if refLen >= 3:    
                    first = word[0]
                    last = word[-1]
                    mid = word[1:-1]
                elif refLen == 2:
                    first = word[0]
                    last = word[-1]
                else:
                    first = word[0]

                charTups.append( (first, mid, last, refLen) )
                
            conll_data.append( ( annotations[input_idx], charTups) )
            
            #conll_data.append( annotations[input_idx] )
            logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
        else:
            assert(label_idx < len(annotations))
            
            charTups = []
            for word in annotations[input_idx]:
        
                first = mid = last = None
                refLen = len(word)
                if refLen >= 3:    
                    first = word[0]
                    last = word[-1]
                    mid = word[1:-1]
                elif refLen == 2:
                    first = word[0]
                    last = word[-1]
                else:
                    first = word[0]

                charTups.append( (first, mid, last, refLen) )
                
            conll_data.append( ( annotations[input_idx], annotations[label_idx] , charTups) )
            logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
            
    return conll_data

def prepare_sequence(seq, to_ix, unk):

    if type(seq[0]) == tuple:
        charOHs = []
        for charTup in seq:
            oneHot = np.zeros((3, len(to_ix)))
            # Last item in tuple was saved as word len
            refLen = charTup[-1]
            if refLen >= 3:    
                first = charTup[0]
                mid = charTup[1]
                last = charTup[2] 
                
                # Could be more than one so just add
                for c in mid:
                    if c not in to_ix:
                        c = "unk"
                    oneHot[1, to_ix[c]] += 1.0

                if last not in to_ix:
                    last = "unk"
                oneHot[2, to_ix[last]] += 1.0
            elif refLen == 2:
                first = charTup[0]
                last = charTup[2]
                
                if last not in to_ix:
                    last = "unk"
                    
                oneHot[2, to_ix[last]] += 1.0
            else:
                first = charTup[0]
            
            if first not in to_ix:
                first = "unk"
                
            oneHot[0, to_ix[first]] += 1.0
            
            charOHs.append(oneHot)
        charOHs = np.stack(charOHs)
        charOHs = torch.from_numpy(charOHs).type(torch.FloatTensor)
        charOHs = charOHs.view(-1, charOHs.shape[-1])
        return charOHs
    else:
        idxs = []
        if unk not in to_ix:
            idxs = [to_ix[w] for w in seq]
        else:
            idxs = [to_ix[w] for w in map(lambda w: unk if w not in to_ix else w, seq)]
        return torch.tensor(idxs, dtype=torch.long)

class LSTMTaggerModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, char_size):
        torch.manual_seed(1)
        super(LSTMTaggerModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeddings = nn.Parameter(torch.zeros(char_size, embedding_dim))
        
        torch.nn.init.normal_(self.char_embeddings)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # If sum chars
        # self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, bidirectional=False)
        # If 4 unique embeds
        self.lstm = nn.LSTM(embedding_dim * 4, hidden_dim, bidirectional=False)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, charEmbed):
        embeds = self.word_embeddings(sentence)
        #print(embeds.shape)
        # Not sure what embedding dim is, whether [B, Max Set Len, Embed dim] or something else???
        charEmbeds = torch.matmul(charEmbed, self.char_embeddings)
        #print(charEmbeds.shape)
        #charEmbeds = charEmbeds.view(int(charEmbeds.shape[0] / 3), 3, charEmbeds.shape[1]).sum(1)
        
        # Concat all this time
        charEmbeds = charEmbeds.view(int(charEmbeds.shape[0] / 3), 3, charEmbeds.shape[1])
        charEmbeds = charEmbeds.view(charEmbeds.shape[0], -1)
        
        #print(charEmbeds.shape)
        embeds = torch.cat([embeds, charEmbeds],-1)
        #print(embeds.shape)
        #print("Concat Done")
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class LSTMTagger:

    def __init__(self, trainfile, modelfile, modelsuffix, unk="[UNK]", epochs=10, embedding_dim=128, hidden_dim=64):
        self.unk = unk
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.training_data = []
        if trainfile[-3:] == '.gz':
            with gzip.open(trainfile, 'rt') as f:
                self.training_data = read_conll(f)
        else:
            with open(trainfile, 'r') as f:
                self.training_data = read_conll(f)

        self.word_to_ix = {} # replaces words with an index (one-hot vector)
        self.tag_to_ix = {} # replace output labels / tags with an index
        self.ix_to_tag = [] # during inference we produce tag indices so we have to map it back to a tag

        chars = set()
        for sent, tags, charTups in self.training_data:
            
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                    
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag.append(tag)
                    
            for ct in charTups:
    
                chars.add(ct[0])

                if ct[2] is not None:
                    chars.add(ct[2])

                if ct[1] is not None:
                    for char in ct[1]:
                        chars.add(char)

        chars = list(chars)
        chars = sorted(chars)
        charToDex = {char: dex for dex, char in enumerate(chars)}
        charToDex['unk'] = len(charToDex)
        self.charToDex = charToDex
        
        logging.info("word_to_ix:", self.word_to_ix)
        logging.info("tag_to_ix:", self.tag_to_ix)
        logging.info("ix_to_tag:", self.ix_to_tag)
        logging.info("char_to_dex:", self.charToDex)
        
        #print("Creating Modified Model")
        self.model = LSTMTaggerModel(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), len(self.tag_to_ix), len(self.charToDex))
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def argmax(self, seq, charTups):
        output = []
        with torch.no_grad():
            inputs = prepare_sequence(seq, self.word_to_ix, self.unk)
            charEmbeds = prepare_sequence(charTups, self.charToDex, "unk")
            tag_scores = self.model(inputs, charEmbeds)
            for i in range(len(inputs)):
                output.append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
        return output

    def train(self):
        loss_function = nn.NLLLoss()

        self.model.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            for sentence, tags, charTups in tqdm.tqdm(self.training_data):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = prepare_sequence(sentence, self.word_to_ix, self.unk)
                targets = prepare_sequence(tags, self.tag_to_ix, self.unk)
                charEmbeds = prepare_sequence(charTups, self.charToDex, "unk")

                # Step 3. Run our forward pass.
                tag_scores = self.model(sentence_in, charEmbeds)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
                loss.backward()
                self.optimizer.step()

            if epoch == self.epochs-1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            print("saving model file: {}".format(savefile), file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'unk': self.unk,
                        'word_to_ix': self.word_to_ix,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                        'char_to_dex': self.charToDex
                    }, savefile)

    def decode(self, inputfile):
        if inputfile[-3:] == '.gz':
            with gzip.open(inputfile, 'rt') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)
        else:
            with open(inputfile, 'r') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)

        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError("Error: missing model file {}".format(self.modelfile + self.modelsuffix))

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        epoch = saved_model['epoch']
        loss = saved_model['loss']
        self.unk = saved_model['unk']
        self.word_to_ix = saved_model['word_to_ix']
        self.tag_to_ix = saved_model['tag_to_ix']
        self.ix_to_tag = saved_model['ix_to_tag']
        self.charToDex = saved_model['char_to_dex']
        self.model.eval()
        #print("Decoding")
        decoder_output = []
        for sent, charTups in tqdm.tqdm(input_data):
            #print(sent)
            decoder_output.append(self.argmax(sent, charTups))
        return decoder_output

if __name__ == '__main__':
    #print("Chunk Opts")
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join('data', 'input', 'dev.txt'), help="produce chunking output for this input file")
    optparser.add_option("-t", "--trainfile", dest="trainfile", default=os.path.join('data', 'train.txt.gz'), help="training data for chunker")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join('data', 'charEmbedMod'), help="filename without suffix for model files")
    optparser.add_option("-s", "--modelsuffix", dest="modelsuffix", default='.tar', help="filename suffix for model files")
    optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="number of epochs [fix at 5]")
    optparser.add_option("-u", "--unknowntoken", dest="unk", default='[UNK]', help="unknown word token")
    optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can be slow)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    #print("Chunker Start")
    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    modelfile = opts.modelfile
    if opts.modelfile[-4:] == '.tar':
        modelfile = opts.modelfile[:-4]
    #print("Creating Chunker")
    chunker = LSTMTagger(opts.trainfile, modelfile, opts.modelsuffix, opts.unk)
    # use the model file if available and opts.force is False
    if os.path.isfile(opts.modelfile + opts.modelsuffix) and not opts.force:
        decoder_output = chunker.decode(opts.inputfile)
    else:
        print("Warning: could not find modelfile {}. Starting training.".format(modelfile + opts.modelsuffix), file=sys.stderr)
        chunker.train()
        decoder_output = chunker.decode(opts.inputfile)

    print("\n\n".join([ "\n".join(output) for output in decoder_output ]))
