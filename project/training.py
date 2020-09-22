import os
from read_files import split_imdb_files, split_yahoo_files, split_agnews_files
from word_level_process import word_process, get_tokenizer
from char_level_process import char_process
from neural_networks import word_cnn, char_cnn, bd_lstm, lstm
import keras
from keras import backend as K
import tensorflow as tf
import argparse
from config import config
from sklearn.utils import shuffle
import pdb
from config import config
from keras import losses
import scipy
import math


tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
K.set_session(tf.Session(config=tf_config))

parser = argparse.ArgumentParser(
    description='Train a text classifier.')
parser.add_argument('-m', '--model',
                    help='The model of text classifier',
                    choices=['word_cnn', 'char_cnn', 'word_lstm', 'word_bdlstm'],
                    default='word_cnn')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews', 'yahoo'],
                    default='imdb')
parser.add_argument('-l', '--level',
                    help='The level of process dataset',
                    choices=['word', 'char'],
                    default='word')


def train_text_classifier():
    dataset = args.dataset
    x_train = y_train = x_test = y_test = None
    if dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'yahoo':
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    # Take a look at the shapes
    print('dataset:', dataset, '; model:', args.model, '; level:', args.level)
    print('X_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', x_test.shape)
    print('y_test:', y_test.shape)

    log_dir = r'./logs/{}/{}/'.format(dataset, args.model)
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)

    model_path = r'./runs/{}/{}.dat_100embed_binary'.format(dataset, args.model)
    model = batch_size = epochs = None
    assert args.model[:4] == args.level

    if args.model == "word_cnn":
        model = word_cnn(dataset)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.wordCNN_epochs[dataset]
    elif args.model == "word_bdlstm":
        model = bd_lstm(dataset)
        batch_size = config.bdLSTM_batch_size[dataset]
        epochs = config.bdLSTM_epochs[dataset]
    elif args.model == "char_cnn":
        model = char_cnn(dataset)
        batch_size = config.charCNN_batch_size[dataset]
        epochs = config.charCNN_epochs[dataset]
    elif args.model == "word_lstm":
        model = lstm(dataset)
        batch_size = config.LSTM_batch_size[dataset]
        epochs = config.LSTM_epochs[dataset]

    print('Train...')
    print('batch_size: ', batch_size, "; epochs: ", epochs)


    # get threshold
    ## compute Gilbert–Varshamov Bound ##
    def _check_(L, M, d):
        left = 0
        for i in range(0, d):
            left = left + scipy.special.binom(L,i)
        tmp = float(pow(2, int(L))) / left
        return (tmp>M)

    def gvbound(L,M,tight=False):
        d = 1
        while (_check_(L, M, d)): d += 1
        if not tight: d -= 1
        print('Distance for %d classes with %d bits (GV bound): %d' %(M,L,d))
        return d

    ## compute Hamming Bound ##
    def check(L, M, d):
        right = float(pow(2, int(L))) / float(M)
        x = math.floor(float(d - 1) / 2.0)
        left = 0
        for i in range(0, x+1):
            left = left + scipy.special.binom(L,i)
        return (left <= right)


    def hammingbound(L,M,tight=False):
        d = 0
        while (check(L, M, d)): d += 1
        if not tight: d -= 1
        print('Distance for %d classes with %d bits (Hamming bound): %d' %(M,L,d))
        return d
    embedding_dims = config.wordCNN_embedding_dims[dataset]
    num_words = config.num_words[dataset]
    ham_bound = hammingbound(embedding_dims, num_words)
    gv_bound = gvbound(embedding_dims,num_words)
    bound = ham_bound

    def custom_loss_wrapper(y_true, y_pred):
        # ce-loss
        xent_loss = losses.binary_crossentropy(y_true, y_pred)
        # quantization loss
        embedding_output = model.layers[0].output
        Bbatch = tf.sign(embedding_output)
        quan_l = K.mean(keras.losses.mean_absolute_error(Bbatch,embedding_output))
        # ecoc loss
        centers = model.layers[0].weights[0]
        centers_binary = tf.sign(centers)
        norm_centers = tf.math.l2_normalize(centers, axis=1)
        prod = tf.matmul(norm_centers, tf.transpose(norm_centers)) #5000 x 5000
        ham_dist = 0.5 * (embedding_dims - prod)
        marg_loss = K.mean(ham_dist)
        #tmp = K.maximum(0.0, ham_dist - bound)
        #non_zero = K.sum(K.cast(tmp > 0, dtype='float32'))
        #marg_loss = K.sum(tmp) * 1.0/(non_zero+1.0)

        #tmp2 = K.maximum(0.0, gv_bound - ham_dist)
        #non_zero2 = K.sum(K.cast(tmp2 > 0, dtype='float32'))
        #marg_loss2 = K.sum(tmp2) * 1.0/(non_zero2+1.0)
        
        return  xent_loss + 0.1*quan_l

       
    #centers = model.layers[0].weights[0]
    #norm_centers = tf.math.l2_normalize(centers, axis=1)
    #prod = tf.matmul(norm_centers, tf.transpose(norm_centers))
    #ham_dist = 0.5 * (embedding_dims - prod)
    #marg_loss = K.mean(K.maximum(0.0, 26 - ham_dist))
    #get_output = K.function([model.layers[0].input],
                            #[prod])
    #x = get_output([x_train])[0]

 

    #centers = centers[0]
    #prod = tf.matmul(centers, tf.transpose(centers))

    #pdb.set_trace()
    

    def quan_loss(y_true, y_pred):
        embedding_output = model.layers[0].output
        Bbatch = tf.sign(embedding_output)
        quan_l = K.mean(keras.losses.mean_absolute_error(Bbatch,embedding_output))
        return  quan_l

    def ecoc_loss(y_true, y_pred):
        # ecoc loss
        centers = model.layers[0].weights[0]
        centers_binary = tf.sign(centers)
        norm_centers = tf.math.l2_normalize(centers, axis=1)
        prod = tf.matmul(norm_centers, tf.transpose(norm_centers)) #5000 x 5000
        #ham_dist = 0.5 * (embedding_dims - prod)
        marg_loss = K.mean(prod)
        #tmp = K.maximum(0.0, bound - ham_dist)
        #non_zero = K.sum(K.cast(tmp > 0, dtype='float32'))
        #marg_loss = K.sum(tmp) * 1.0/(non_zero+1.0)
        #tmp = K.maximum(0.0, ham_dist - bound)
        #non_zero = K.sum(K.cast(tmp > 0, dtype='float32'))
        #marg_loss = K.sum(tmp) * 1.0/(non_zero+1.0)
        return  marg_loss

    model.compile(loss=custom_loss_wrapper,
                  optimizer='adam',
                  metrics=['accuracy',quan_loss,ecoc_loss])

    #tmp = model.predict(x_train,batch_size=batch_size)

    #pdb.set_trace()
    model.fit(x_train, y_train,
              batch_size=batch_size,  # batch_size => 32
              epochs=10,#epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tb_callback])
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    print('Saving model weights...')
    model.save_weights(model_path)


if __name__ == '__main__':
    args = parser.parse_args()
    train_text_classifier()
