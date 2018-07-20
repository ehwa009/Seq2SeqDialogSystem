import tensorflow as tf
import random
import math
import os

from model import Seq2Seq
from dialog import Dialog
from config import FLAGS

def train(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.vocab_size)
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("read model from existed one")
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("creating new model")
            sess.run(tf.global_variables_initializer())

        # writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        total_batch = int(math.ceil(len(dialog.examples)/float(batch_size)))

        for step in range(total_batch * epoch):
            enc_input, dec_input, targets = dialog.next_batch(batch_size)

            _, loss = model.train(sess, enc_input, dec_input, targets)
            
            if(step+1)%100 == 0:
                print('Step:', '%06d' % model.global_step.eval(),'Cost =', '{:.6f}'.format(loss))
        
        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print("training complete.")

def test(dialog, batch_size=100):
    pass


def main(_):
    dialog = Dialog()

    dialog.load_vocab(FLAGS.voc_path)
    dialog.load_examples(FLAGS.data_path)

    if FLAGS.train:
        train(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
    elif FLAGS.test:
        pass

if __name__ == "__main__":
    tf.app.run()