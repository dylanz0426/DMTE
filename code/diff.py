import numpy as np
import tensorflow as tf
import config

alpha = 0.3
beta = 0.1

class Model:
    def __init__(self, vocab_size, num_nodes):
        # '''hyperparameter'''
        with tf.name_scope('read_inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Ta')
            self.Text_b = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tb')
            self.Text_neg = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tneg')
            self.Node_a = tf.placeholder(tf.int32, [config.batch_size], name='n1')
            self.Node_b = tf.placeholder(tf.int32, [config.batch_size], name='n2')
            self.Node_neg = tf.placeholder(tf.int32, [config.batch_size], name='n3')
            self.P_a = tf.placeholder(tf.float32, [config.batch_size, config.batch_size], name='Pa')
            self.P_b = tf.placeholder(tf.float32, [config.batch_size, config.batch_size], name='Pb')
            self.P_neg = tf.placeholder(tf.float32, [config.batch_size, config.batch_size], name='Pneg')

        with tf.name_scope('initialize_embedding') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, config.word_embed_size], stddev=0.3))
            self.node_embed = tf.Variable(tf.truncated_normal([num_nodes, config.embed_size / 2], stddev=0.3))
            self.node_embed = tf.clip_by_norm(self.node_embed, clip_norm=1, axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            self.TA = tf.nn.embedding_lookup(self.text_embed, self.Text_a)
            self.T_A = tf.expand_dims(self.TA, -1)

            self.TB = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            self.T_B = tf.expand_dims(self.TB, -1)

            self.TNEG = tf.nn.embedding_lookup(self.text_embed, self.Text_neg)
            self.T_NEG = tf.expand_dims(self.TNEG, -1)

            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)
            self.N_NEG = tf.nn.embedding_lookup(self.node_embed, self.Node_neg)
        self.convA, self.convB, self.convNeg = self.conv()
        self.loss = self.compute_loss()

    def conv(self):
        
        W0 = tf.Variable(tf.truncated_normal([config.word_embed_size, config.embed_size / 2], stddev=0.3))
        W1 = tf.Variable(tf.truncated_normal([config.word_embed_size, config.embed_size / 2], stddev=0.3))
        W2 = tf.Variable(tf.truncated_normal([config.word_embed_size, config.embed_size / 2], stddev=0.3))
        
        mA = tf.reduce_mean(self.T_A, axis=1, keepdims=True)
        mB = tf.reduce_mean(self.T_B, axis=1, keepdims=True)
        mNEG = tf.reduce_mean(self.T_NEG, axis=1, keepdims=True)
        
        convA = tf.tanh(tf.squeeze(mA))
        convB = tf.tanh(tf.squeeze(mB))
        convNEG = tf.tanh(tf.squeeze(mNEG))
        
        attA = tf.matmul(convA, W0) + alpha * tf.matmul(tf.matmul(self.P_a, convA), W1) + beta * tf.matmul(tf.matmul(tf.square(self.P_a), convA), W2)
        attB = tf.matmul(convB, W0) + alpha * tf.matmul(tf.matmul(self.P_b, convB), W1) + beta * tf.matmul(tf.matmul(tf.square(self.P_b), convB), W2)
        attNEG = tf.matmul(convNEG, W0) + alpha * tf.matmul(tf.matmul(self.P_a, convNEG), W1) + beta * tf.matmul(tf.matmul(tf.square(self.P_a), convNEG), W2)
        
        return attA, attB, attNEG

    def compute_loss(self):
        
        p1 = tf.reduce_sum(tf.multiply(self.convA, self.convB), 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.001)

        p2 = tf.reduce_sum(tf.multiply(self.convA, self.convNeg), 1)
        p2 = tf.log(tf.sigmoid(-p2) + 0.001)

        p3 = tf.reduce_sum(tf.multiply(self.N_A + alpha * tf.matmul(self.P_a, self.N_A) + beta * tf.matmul(tf.square(self.P_a), self.N_A), self.N_B), 1)
        p3 = tf.log(tf.sigmoid(p3) + 0.001)

        p4 = tf.reduce_sum(tf.multiply(self.N_A + alpha * tf.matmul(self.P_a, self.N_A) + beta * tf.matmul(tf.square(self.P_a), self.N_A), self.N_NEG), 1)
        p4 = tf.log(tf.sigmoid(-p4) + 0.001)

        p5 = tf.reduce_sum(tf.multiply(self.N_A + alpha * tf.matmul(self.P_a, self.N_A) + beta * tf.matmul(tf.square(self.P_a), self.N_A), self.convB), 1)
        p5 = tf.log(tf.sigmoid(p5) + 0.001)

        p6 = tf.reduce_sum(tf.multiply(self.N_A + alpha * tf.matmul(self.P_a, self.N_A) + beta * tf.matmul(tf.square(self.P_a), self.N_A), self.convNeg), 1)
        p6 = tf.log(tf.sigmoid(-p6) + 0.001)

        p7 = tf.reduce_sum(tf.multiply(self.convA, self.N_B), 1)
        p7 = tf.log(tf.sigmoid(p7) + 0.001)

        p8 = tf.reduce_sum(tf.multiply(self.convA, self.N_NEG), 1)
        p8 = tf.log(tf.sigmoid(-p8) + 0.001)
        
        p11 = tf.reduce_sum(tf.multiply(self.convB, self.convA), 1)
        p11 = tf.log(tf.sigmoid(p11) + 0.001)

        p12 = tf.reduce_sum(tf.multiply(self.convB, self.convNeg), 1)
        p12 = tf.log(tf.sigmoid(-p12) + 0.001)

        p13 = tf.reduce_sum(tf.multiply(self.N_B + alpha * tf.matmul(self.P_b, self.N_B) + beta * tf.matmul(tf.square(self.P_b), self.N_B), self.N_A), 1)
        p13 = tf.log(tf.sigmoid(p13) + 0.001)

        p14 = tf.reduce_sum(tf.multiply(self.N_B + alpha * tf.matmul(self.P_b, self.N_B) + beta * tf.matmul(tf.square(self.P_b), self.N_B), self.N_NEG), 1)
        p14 = tf.log(tf.sigmoid(-p14) + 0.001)

        p15 = tf.reduce_sum(tf.multiply(self.N_B + alpha * tf.matmul(self.P_b, self.N_B) + beta * tf.matmul(tf.square(self.P_b), self.N_B), self.convA), 1)
        p15 = tf.log(tf.sigmoid(p15) + 0.001)

        p16 = tf.reduce_sum(tf.multiply(self.N_B + alpha * tf.matmul(self.P_b, self.N_B) + beta * tf.matmul(tf.square(self.P_b), self.N_B), self.convNeg), 1)
        p16 = tf.log(tf.sigmoid(-p16) + 0.001)

        p17 = tf.reduce_sum(tf.multiply(self.convB, self.N_A), 1)
        p17 = tf.log(tf.sigmoid(p17) + 0.001)

        p18 = tf.reduce_sum(tf.multiply(self.convB, self.N_NEG), 1)
        p18 = tf.log(tf.sigmoid(-p18) + 0.001)
        
        rho1 = 0.7
        rho2 = 1.0
        rho3 = 0.1
        temp_loss = rho1 * (p1 + p2 + p11 + p12) + rho2 * (p3 + p4 + p13 + p14) + rho3 * (p5 + p6 + p15 + p16) + rho3 * (p7 + p8 + p17 + p18)
        loss = -tf.reduce_sum(temp_loss)
        return loss
