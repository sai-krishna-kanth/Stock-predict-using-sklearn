import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

if __name__ == '__Stock Predict__':


    dataset_train = pd.read_csv('AMZN_TRAIN.csv')
    dataset_test = pd.read_csv('AMZN_TEST.csv')

    timestamps = 15
    node_l2 = 30
    node_l3 = 11
    node_output = 1
    epochs = 100
    batch_size = 32

    training_set = dataset_train.iloc[:, 1:2].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)


    X_train = []
    y_train = []

    X_train = []
    y_train = []
    for i in range(timestamps, len(training_set)):
        X_train.append(training_set_scaled[i - timestamps:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    y_train = y_train.reshape(-1, 1)

    real_stock_price = dataset_test.iloc[:, 1:2].values

    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestamps:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    y_test = []
    for i in range(timestamps, timestamps + len(dataset_test)):
        X_test.append(inputs[i - timestamps:i, 0])
        y_test.append(inputs[i, 0])
    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(-1, 1)

    y_t = sc.inverse_transform(y_test)


    def total_accuracy(original_val, predicted_val):

        total_error = 0
        for i in range(0, original_val.shape[0]):
            err = abs(original_val[i] - predicted_val[i]) / original_val[i]
            total_error += err
        print("Error: ", total_error * 100 / original_val.shape[0], " %")
        print("Accuracy: ", 100 - total_error * 100 / original_val.shape[0], " %")


    def neural_net_model(X_data, input_dim):
        W_1 = tf.Variable(tf.random_uniform([input_dim, node_l2]))
        b_1 = tf.Variable(tf.zeros([node_l2]))
        layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
        layer_1 = tf.nn.tanh(layer_1)

        W_2 = tf.Variable(tf.random_uniform([node_l2, node_l3]))
        b_2 = tf.Variable(tf.zeros([node_l3]))
        layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
        layer_2 = tf.nn.tanh(layer_2)

        W_O = tf.Variable(tf.random_uniform([node_l3, node_output]))
        b_O = tf.Variable(tf.zeros([1]))
        output = tf.add(tf.matmul(layer_2, W_O), b_O)

        return output, W_O


    xs = tf.placeholder("float")
    ys = tf.placeholder("float")

    output, W_O = neural_net_model(xs, timestamps)

    cost = tf.reduce_mean(tf.square(output - ys))
    train = tf.train.AdamOptimizer(0.001).minimize(cost)

    correct_pred = tf.argmax(output, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    c_t = []
    c_test = []

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

        for i in range(epochs):
            for j in range(X_train.shape[0]):
                sess.run([cost, train], feed_dict={xs: X_train[j, :].reshape(1, -1), ys: y_train[j][0]})


            c_t.append(sess.run(cost, feed_dict={xs: X_train, ys: y_train}))
            c_test.append(sess.run(cost, feed_dict={xs: X_test, ys: y_test}))
            print('Epoch :', i, 'Cost :', c_t[i])

        pred = sess.run(output, feed_dict={xs: X_test})
        pred = pred.reshape(-1, 1)

        print('Cost :', sess.run(cost, feed_dict={xs: X_test, ys: y_test}))

        pred = sc.inverse_transform(pred)

    import math
    from sklearn.metrics import mean_squared_error

    rmse = math.sqrt(mean_squared_error(y_t, pred))
    print("rmse=", rmse)
    # Visualising the results
    plt.plot(y_t, color='red', label='Real Amazon Stock Price')
    plt.plot(pred, color='blue', label='Predicted Amazon Stock Price')
    plt.title('Amazon Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amazon Stock Price')
    plt.legend()
    plt.savefig("graph_amazon.png")
    plt.show()

    total_accuracy(y_t, pred)
