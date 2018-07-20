import tensorflow as tf
import numpy as np
import math
import sys

from config import FLAGS
from model import Seq2Seq
from dialog import Dialog

class Chatbot:

    def __init__(self, voc_path, train_dir):
        self.dialog = Dialog()
        self.dialog.load_vocab(voc_path)

        self.model = Seq2Seq(self.dialog.vocab_size)

        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(train_dir)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def run(self):
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = sys.stdin.readline()

        while line:
            pass

    def _decode(self, enc_input, dec_input):
        pass
    
    def _get_replay(self, msg):
        enc_input = self.dialog.tokenizer(msg)
        enc_input = self.dialog.tokens_to_ids(enc_input)
        dec_input = []

        curr_seq = 0
        for i in range(FLAGS.max_decode_len):
            pass

    def main(_):
        print("initialzing dialog system")

        chatbot = Chatbot(FLAGS.voc_path, FLAGS.train_dir)

if __name__ == "__main__":
    tf.app.run()
