import time, datetime
import os
import tensorflow as tf
import numpy as np
from cnn_model import SentenceCNN
import preprocessing

###########################################
# Defining Parameters
###########################################

tf.flags.DEFINE_integer("n_classes", 6, "Number of classes")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("n_filters_per_size", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("lambda_l2_reg", 0.1, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("l2_norm_cutoff", 3.0, "Cutoff for l2 norm regularization of weights (default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("n_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(tf.flags.FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

###########################################
# Data Preparation
###########################################

print("loading data...")
x, y, vocab, vocab_inv = preprocessing.load_data()
# Shuffle data randomly
np.random.seed(10)

# shuffled_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffled_indices]
# y_shuffled = y[shuffled_indices]

# split train/valid set
# TODO Implement a fucking correct XVal procedure for this
x_train, x_valid = x[:-500], x[-500:]
y_train, y_valid = y[:-500], y[-500:]
print("Vocabulary Size: {:d}".format(len(vocab)))
print("Train/Validation split: {:d}/{:d}".format(len(y_train), len(y_valid)))

###########################################
# Training
###########################################

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=tf.flags.FLAGS.allow_soft_placement,
                                  log_device_placement=tf.flags.FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = SentenceCNN(
            seq_len=x_train.shape[1],
            n_classes=tf.flags.FLAGS.n_classes,
            vocab_size=len(vocab),
            embedding_size=tf.flags.FLAGS.embedding_dim,
            filter_sizes=list(map(int, tf.flags.FLAGS.filter_sizes.split(","))),
            n_filters_per_size=tf.flags.FLAGS.n_filters_per_size,
            lambda_l2_reg=tf.flags.FLAGS.lambda_l2_reg)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

        # here we keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.all_variables())

        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            feed_dict = {cnn.x: x_batch, cnn.y: y_batch, cnn.dropout_keep_prob: tf.flags.FLAGS.dropout_keep_prob}
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op,
                                                           cnn.loss, cnn.accuracy], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            # update the weight values after each training step
            W_fc = cnn.W_fc.eval()
            l2_norm = np.linalg.norm(W_fc)
            # print(l2_norm)
            if l2_norm > tf.flags.FLAGS.l2_norm_cutoff:
                # first scale the matrix so the norm is 1 and then multiply by the cutoff
                W_fc /= l2_norm
                W_fc *= tf.flags.FLAGS.l2_norm_cutoff
                assign_op = cnn.W_fc.assign(W_fc)
                sess.run(assign_op)
                W_fc = cnn.W_fc.eval()
                l2_norm = np.linalg.norm(W_fc)
                print(">>>>>>> NEW L2 NORM IS:" + str(l2_norm))

        def validation_step(x_batch, y_batch, writer=None):
            feed_dict = {cnn.x: x_batch, cnn.y: y_batch, cnn.dropout_keep_prob: 1.0}
            step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                                                       feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = preprocessing.batch_iter(list(zip(x_train, y_train)), tf.flags.FLAGS.batch_size,
                                           tf.flags.FLAGS.n_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % tf.flags.FLAGS.evaluate_every == 0:
                print("\nEvaluation on Validation set:")
                validation_step(x_valid, y_valid, writer=dev_summary_writer)
                print("")
            if current_step % tf.flags.FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
