############################################Eg1:  Logistic Regression with Mini-Batch Gradient Descent using TensorFlow


# Common imports
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# to reset the graph to blank 
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#retrieving data (the first step always) 
#total sample number: 1000
m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
y_moons_column_vector = y_moons.reshape(-1, 1)
#visualize the overall dataset
plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
plt.legend()
plt.show()


#splitting data (20% testing, 80% training and no validation set since we dont do any hyperparameter tuning here!)
test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


#graph construction phase
reset_graph()
#two features for each sample 
#the input layer consists of two input neurons each corresponding one feature
#the input neurons do nothing but pass the input features
n_inputs = 2
#construct the placeholders ("the variable box")
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
#initialize the theta (the weights of (features + bias))
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")

#derive the loss function
#define the weighted sum
logits = tf.matmul(X, theta, name="logits")
#define the logistic sigmoid function ***  y(x) = 1 / (1 + tf.exp(-x))  *** followed the weighted sum
y_proba = tf.sigmoid(logits)
epsilon = 1e-7  # to avoid an overflow when computing the log
#define the loss fucntion, which is cross entropy here ***   loss(cross_entropy) = mean(y * log10(y_proba) + (1 - y) * log10(1 - y_proba))
loss = tf.losses.log_loss(y, y_proba)  # uses epsilon = 1e-7 by default


learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)#target function
#build an initializer which will initialize all the tf.variables when running
init = tf.global_variables_initializer()

#these below are all hyperparameters, controlling the training process
n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

#excution phase (training  + testing )
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)

    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})

y_pred = (y_proba_val >= 0.5)

print("The precision score is ",precision_score(y_test, y_pred))

#visualize the prediction results
y_pred_idx = y_pred.reshape(-1) # a 1D array rather than a column vector
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()


###############################################Eg2:  Modified Logistic Regression with Mini-Batch Gradient Descent using TensorFlow 
#extend the features for the database
X_train_enhanced = np.c_[X_train,
                         np.square(X_train[:, 1]),
                         np.square(X_train[:, 2]),
                         X_train[:, 1] ** 3,
                         X_train[:, 2] ** 3]
X_test_enhanced = np.c_[X_test,
                        np.square(X_test[:, 1]),
                        np.square(X_test[:, 2]),
                        X_test[:, 1] ** 3,
                        X_test[:, 2] ** 3]

#graph construction phase
reset_graph()

def logistic_regression(X, y, initializer=None, seed=42, learning_rate=0.01):
    n_inputs_including_bias = int(X.get_shape()[1])
    with tf.name_scope("logistic_regression"):
        with tf.name_scope("model"):
            if initializer is None:
                initializer = tf.random_uniform([n_inputs_including_bias, 1], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name="theta")
            logits = tf.matmul(X, theta, name="logits")
            y_proba = tf.sigmoid(logits)
        with tf.name_scope("train"):
            loss = tf.losses.log_loss(y, y_proba, scope="loss")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
        with tf.name_scope("save"):
            saver = tf.train.Saver()
    return y_proba, loss, training_op, loss_summary, init, saver
from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)
#we use 6 features this time (2 original + 4 extended)
n_inputs = 2 + 4
#the location of the logfile is ./tf_logs/logregrun2018xxxxx/
logdir = log_dir("logreg")

X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y)

#put the file_writer initialization at end of the graph design
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10001
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "tmp/my_logreg_model"

#excution phase (training phase + testing phase)
with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
        file_writer.add_summary(summary_str, epoch)
        if epoch % 500 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))

    saver.save(sess, final_model_path)
    y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
    os.remove(checkpoint_epoch_path)

#prediction phase
y_pred = (y_proba_val >= 0.5)
print("The precision score is ",precision_score(y_test, y_pred))
print("The recall score is ",recall_score(y_test, y_pred))
y_pred_idx = y_pred.reshape(-1) # a 1D array rather than a column vector
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()


################################################Eg3:  (application phase) Restored Logistic Regression with Mini-Batch Gradient Descent using TensorFlow 
#extend the features for the database
X_train_enhanced = np.c_[X_train,
                         np.square(X_train[:, 1]),
                         np.square(X_train[:, 2]),
                         X_train[:, 1] ** 3,
                         X_train[:, 2] ** 3]
X_test_enhanced = np.c_[X_test,
                        np.square(X_test[:, 1]),
                        np.square(X_test[:, 2]),
                        X_test[:, 1] ** 3,
                        X_test[:, 2] ** 3]
reset_graph()


#we use 6 features this time (2 original + 4 extended)
n_inputs = 2 + 4
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y)


previous_model_path = "tmp/my_logreg_model"

with tf.Session() as sess:
    #load the previous built trained model
    saver.restore(sess, previous_model_path)

    y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})

#prediction phase
y_pred = (y_proba_val >= 0.5)
print("The precision score is ",precision_score(y_test, y_pred))
print("The recall score is ",recall_score(y_test, y_pred))
y_pred_idx = y_pred.reshape(-1) # a 1D array rather than a column vector
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()