############################################Eg1:  Training an MLP (DNN) with TensorFlow’s High-Level API (no graph!)


# Common imports
import numpy as np
import tensorflow as tf
# To plot pretty figures
import matplotlib.pyplot as plt
from datetime import datetime
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#retrieving data (the first step always) 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

#training phase
feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10,
                                     feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
dnn_clf.train(input_fn=input_fn)

#testing phase
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
print("The prediction result is : {0:.2f}%".format(100*eval_results['accuracy']))
y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
y_pred = list(y_pred_iter)
y_pred[0]



############################################Eg2:  Training an MLP (DNN) with plain Tensorflow
#graph construction phase

reset_graph()

#setting parameters (not to be changed!)
n_inputs = 28*28  # MNIST feature number
n_outputs = 10 # class number

#these below are all the hyperparameters, which controll the training process
n_hidden1 = 300
n_hidden2 = 100
n_epochs = 40
batch_size = 50


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

#define each layer
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

#define the overall topology
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)
    
#define the lost function (here  we use cross_entropy)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)
    
#based on the loss function, define training algorithm (the tf autodiff function is used for the training process)
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#define the evaluation process after the training   
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(y_proba, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#build an initializer which will initialize all the tf.variables when running  
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

logdir = log_dir("mnist_dnn")

#put the file_writer initialization at end of the graph design
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
final_model_path = "tmp/my_deep_mnist_model"

#execution phase
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_summary_str = sess.run(loss_summary, feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(loss_summary_str, epoch)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("For the ",epoch,"th epoch, ", "Training score:", acc_train, ", Val score:", acc_val)
        saver.save(sess, final_model_path)

#prediction phase               
with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})

accuracy_val

############################################Eg3:  Training an MLP (DNN) with plain Tensorflow, adding early stop
#graph construction phase
reset_graph()

#setting parameters (not to be changed!)
n_inputs = 28*28  # MNIST feature number
n_outputs = 10 # class number

#these below are all the hyperparameters, which controll the training process
n_hidden1 = 300
n_hidden2 = 100
n_epochs = 10001
batch_size = 50


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

#define each layer
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

#define the overall topology        
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)
    
#define the lost function    
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

#based on the loss function, define training algorithm (the tf autodiff function is used for the training process)    
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
#define the evaluation process after the training    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(y_proba, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#build an initializer which will initialize all the tf.variables when running     
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

logdir = log_dir("mnist_dnn")

#put the file_writer initialization at end of the graph design
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
m, n = X_train.shape


checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
final_model_path = "tmp/my_deep_mnist_model"

#extra hyperparameters for the early-stop
best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 20

#execution phase
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val, loss_val,  loss_summary_str = sess.run([accuracy, loss, loss_summary], feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(loss_summary_str, epoch)
        if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)            
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

#prediction phase               
with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})

accuracy_val
