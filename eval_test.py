import tensorflow as tf
import preprocessing
import numpy as np

# Eval params
tf.flags.DEFINE_integer("batch_size", 50, "Batch size default to 64")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc params
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load my own test data here
_, _, vocab, vocab_inv = preprocessing.load_data(is_train=True)
x_test, y_test, _, _ = preprocessing.load_data(is_train=False)
y_test = np.argmax(y_test, axis=1)
print("Vocabulary size: {:d}".format(len(vocab)))
print("Test set size: {:d}".format(len(y_test)))

print("\nEvaluating on test data..\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        x = graph.get_operation_by_name("X").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we wanna evaluate
        predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

        # Generate batches for one epoch
        batches = preprocessing.batch_iter(x_test, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_pred = sess.run(predictions, {x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_pred])

correct_pred = float(sum(all_predictions == y_test))
print("Total number of text examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_pred/float(len(y_test))))

