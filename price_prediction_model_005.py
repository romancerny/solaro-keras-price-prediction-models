'''
To Run:

// TERMINAL 1
cd ~/Downloads/gmonitor-master/src
./gmonitor -d 0 -r 1

// TERMINAL 2
cd ~/my_git_repos/solaro-keras-price-prediction-models/
tensorboard --logdir=./

// TERMINAL 3
cd ~/my_git_repos/solaro-keras-price-prediction-models/
source ~/venvs/3.6TF_1.12/bin/activate
python price_prediction_model_005.py 

'''


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import datetime                                         # Import datetime to operate on dates in the data set
import time                                             # time will help us to get current system date and time
import matplotlib.pyplot as plt                         # Matplotlib for creating plots
plt.rcParams["figure.figsize"] = [15, 10]               # Intialize the figure size so that we get large plots to visualize well
import pandas as pd                                     # We will use pandas to load and clean the data set

from keras.layers import Activation, Dense              # Keras will be used to create LSTM network we will be using tensorflow backend in Keras 
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

from mpl_toolkits.axes_grid1.inset_locator import mark_inset    # MPL tool kit will help us to create special plots we will see there use in the end
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

import numpy as np                                      # Finally numpy for algebric calculations 

from keras.callbacks import LearningRateScheduler
import keras.backend as kbck

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from keras.utils import to_categorical


#   _                    _       _       _        
#  | |    ___   __ _  __| |   __| | __ _| |_ __ _ 
#  | |   / _ \ / _` |/ _` |  / _` |/ _` | __/ _` |
#  | |__| (_) | (_| | (_| | | (_| | (_| | || (_| |
#  |_____\___/ \__,_|\__,_|  \__,_|\__,_|\__\__,_|


prefix = "EURUSD_FT_001_"

def load_normalized_data():                                            

    X = pd.read_csv("../../net_core_projects/fingerTrap/" + prefix + "norm_X.csv")
    Y = pd.read_csv("../../net_core_projects/fingerTrap/" + prefix + "labels_Y.csv")

    return X, Y 
    



print("\nloading data...")
X, Y = load_normalized_data()

# print(X.head())
# print(Y.head())


#    ____                _         _             _       _                        _   
#   / ___|_ __ ___  __ _| |_ ___  | |_ _ __ __ _(_)_ __ (_)_ __   __ _   ___  ___| |_ 
#  | |   | '__/ _ \/ _` | __/ _ \ | __| '__/ _` | | '_ \| | '_ \ / _` | / __|/ _ \ __|
#  | |___| | |  __/ (_| | ||  __/ | |_| | | (_| | | | | | | | | | (_| | \__ \  __/ |_ 
#   \____|_|  \___|\__,_|\__\___|  \__|_|  \__,_|_|_| |_|_|_| |_|\__, | |___/\___|\__|
#                                                                |___/                


SPLIT      = 0.8 # data split ration for training and testing

X = X.values.astype("float32")
Y = Y.values.astype("float32")

train_size = int(len(X) * SPLIT)
X_train = X[:train_size]
Y_train = Y[:train_size]

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))



#    ____                _         _            _              _   
#   / ___|_ __ ___  __ _| |_ ___  | |_ ___  ___| |_   ___  ___| |_ 
#  | |   | '__/ _ \/ _` | __/ _ \ | __/ _ \/ __| __| / __|/ _ \ __|
#  | |___| | |  __/ (_| | ||  __/ | ||  __/\__ \ |_  \__ \  __/ |_ 
#   \____|_|  \___|\__,_|\__\___|  \__\___||___/\__| |___/\___|\__|


X_test = X[train_size:]
Y_test = Y[train_size:]

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))



print("\nX_train")
print(X_train.shape[0], " by ", X_train.shape[1], " by ", X_train.shape[2])                                                                      
print(X_train[0])                                                                    
print("\nY_train")
print(Y_train.shape[0], " by ", Y_train.shape[1])                                                                      
print(Y_train[0])                                                                     
print("\nX_test")
print(X_test.shape[0], " by ", X_test.shape[1], " by ", X_test.shape[2])                                                                      
print(X_test[0])                                                                     
print("\nY_test")
print(Y_test.shape[0], " by ", Y_test.shape[1])                                                                      
print(Y_test[0])                                                                   



#    ____                _                             _      _ 
#   / ___|_ __ ___  __ _| |_ ___   _ __ ___   ___   __| | ___| |
#  | |   | '__/ _ \/ _` | __/ _ \ | '_ ` _ \ / _ \ / _` |/ _ \ |
#  | |___| | |  __/ (_| | ||  __/ | | | | | | (_) | (_| |  __/ |
#   \____|_|  \___|\__,_|\__\___| |_| |_| |_|\___/ \__,_|\___|_|

def build_model(inputs, dropout=0.25, weights_path=None): 
    '''
    https://keras.io/getting-started/sequential-model-guide/
    '''
    
    ####################### 1layer2048_16 ############## batch_size=8192 #################   11s - loss: 0.1764     (on 600 epochs)
    # model = Sequential()
    # model.add(LSTM(2048, input_shape=(inputs.shape[1], inputs.shape[2])))
    # model.add(Dropout(dropout))
    # model.add(Dense(16, kernel_initializer="uniform", activation="relu"))
    # model.add(Dense(3, kernel_initializer="uniform", activation="relu")) 


    ####################### 1layer2048_16 ############## batch_size=128 #################   57s - loss: 0.1873 - acc: 0.9252  (on 191 epochs) tensorboard - 1543517793.521582
    # model = Sequential()
    # model.add(LSTM(2048, input_shape=(inputs.shape[1], inputs.shape[2])))
    # model.add(Dropout(dropout))
    # model.add(Dense(16, kernel_initializer="uniform", activation="relu"))
    # model.add(Dense(3, kernel_initializer="uniform", activation="softmax"))    
    

    ####################### 1layer1024_16 ############## batch_size=128 #################    - 26s - loss: 0.1387 - acc: 0.9463  (on 800 epochs) tensorboard - 1543528948.2887285
    # model = Sequential()
    # model.add(LSTM(1024, input_shape=(inputs.shape[1], inputs.shape[2])))
    # model.add(Dropout(dropout))
    # model.add(Dense(16, kernel_initializer="uniform", activation="relu"))
    # model.add(Dense(3, kernel_initializer="uniform", activation="softmax"))    
    
    ####################### 3layer2048_1024_512_16 ############## batch_size=128 #################   - 86s - loss: 6.7920e-05 - acc: 1.0000   (on 300 epochs) tensorboard - 1543653002.0073202 
    # WINNER !!!
    #model = Sequential()
    #model.add(LSTM(2048, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))
    #model.add(LSTM(1024, return_sequences=True))
    #model.add(LSTM(512, return_sequences=False))
    #model.add(Dropout(dropout))
    #model.add(Dense(16, kernel_initializer="uniform", activation="relu"))
    #model.add(Dense(3, kernel_initializer="uniform", activation="softmax"))    
    
    ####################### 3layer2048_1024_512_16 ############## batch_size=128 #################   - 86s - loss: 6.7920e-05 - acc: 1.0000   (on 300 epochs) tensorboard - 1543653002.0073202 
    # WINNER !!!
    model = Sequential()
    model.add(LSTM(2048, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))
    model.add(LSTM(1024, return_sequences=True))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(16, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))    
    
    ####################### 4layer1024_512_256_128_16 ############## batch_size=128 #################      (on 20 epochs seems same as 1layer2048_1024_512_16) tensorboard - 1543618776.6354616
    # model = Sequential()
    # model.add(LSTM(1024, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(LSTM(256, return_sequences=True))
    # model.add(LSTM(128, return_sequences=False))
    # model.add(Dropout(dropout))
    # model.add(Dense(16, kernel_initializer="uniform", activation="relu"))
    # model.add(Dense(3, kernel_initializer="uniform", activation="softmax"))    
    
    ####################### 4layer2048_1024_512_256_16 ############## batch_size=128 #################   - 89s - loss: 4.0958e-07 - acc: 1.0000   (on 370 epochs) tensorboard - 1543619740.860065
    # model = Sequential()
    # model.add(LSTM(2048, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))
    # model.add(LSTM(1024, return_sequences=True))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(LSTM(256, return_sequences=False))
    # model.add(Dropout(dropout))
    # model.add(Dense(16, kernel_initializer="uniform", activation="relu"))
    # model.add(Dense(3, kernel_initializer="uniform", activation="softmax"))    
    



# TODO try 1024, then deeper layers
# https://stackoverflow.com/a/43944251
# recurrent_dropout

    
    if weights_path is not None:                                                                    # If you have already train weights it can load them
        model.load_weights(weights_path)
    
    return model                                                                                 


np.random.seed(202)                                                                                 # random seed for reproducibility

print("\nbuilding model...")
#bt_model = build_model(X_train)                                                                    
bt_model = build_model(X_train, dropout=0.25)                                                                    
# bt_model = build_model(X_train, weights_path="FxcmPrediction.h5")

print(bt_model.summary())


print("\ncompiling model...")
# bt_model.compile(loss="mae", optimizer="adam", metrics=['accuracy'])    
#bt_model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])    
bt_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])    


#   _____          _                             _      _ 
#  |_   _| __ __ _(_)_ __    _ __ ___   ___   __| | ___| |
#    | || '__/ _` | | '_ \  | '_ ` _ \ / _ \ / _` |/ _ \ |
#    | || | | (_| | | | | | | | | | | | (_) | (_| |  __/ |
#    |_||_|  \__,_|_|_| |_| |_| |_| |_|\___/ \__,_|\___|_|

print("\nfitting model...")


def scheduler(epoch):

    if epoch%10==0 and epoch!=0:

        lr = kbck.get_value(bt_model.optimizer.lr)

        kbck.set_value(bt_model.optimizer.lr, lr*.9)
        print("lr changed to {}".format(lr*.9))

    return kbck.get_value(bt_model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
callbacks_list = [checkpoint, tensorboard, lr_decay]



#################################################################################################### Train model on data
# bt_history = bt_model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2, shuffle=True) 
# bt_history = bt_model.fit(X_train, Y_train, epochs=100, batch_size=4096, callbacks=callbacks_list, verbose=2, shuffle=True) 
# bt_history = bt_model.fit(X_train, Y_train, epochs=1500, batch_size=8192, callbacks=callbacks_list, verbose=2, shuffle=True)
#bt_history = bt_model.fit(X_train, Y_train, epochs=300, batch_size=128, callbacks=callbacks_list, verbose=2, shuffle=True) 
#bt_history = bt_model.fit(X_train, Y_train, epochs=300, validation_data=(X_test, Y_test), batch_size=256, callbacks=callbacks_list, verbose=2, shuffle=True) 
bt_history = bt_model.fit(X_train, Y_train, epochs=300, validation_data=(X_test, Y_test), batch_size=512, callbacks=callbacks_list, verbose=2, shuffle=False) 


print("\nsaving model...")
bt_model.save(prefix + "FxcmPrediction.h5")                                                               # After training save the model on disk


# list all data in history
print(bt_history.history.keys())


# #~~~~~~~~~~~~~~~~~~~~~~~~~~ charts v1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # plot charts
# history = bt_history.history
# loss_history = history["loss"]
# accuracy_history = history["acc"]
# epochs = bt_history.epoch
# plt.plot(epochs,loss_history)
# plt.plot(epochs,accuracy_history)


# ###### load the best weights
# # bt_model.load_weights("weights.best.hdf5")
# bt_model.load_weights("weights.best_3layer2048_1024_512_16.hdf5")

# # Evaluate your performance in one line:
# loss_and_metrics = bt_model.evaluate(X_test, Y_test, batch_size=128)

# loss_history_eval = loss_and_metrics[0]
# accuracy_history_eval = loss_and_metrics[1]

# print('Test loss:', loss_and_metrics[0])
# print('Test accuracy:', loss_and_metrics[1])
# print('Test accuracy:', loss_and_metrics.epoch)


# plt.show()


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~ charts v2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# plot train and validation loss
plt.plot(bt_history.history['loss'])
plt.plot(bt_history.history['val_loss'])
plt.plot(bt_history.history['acc'])
plt.plot(bt_history.history['val_acc'])
plt.title('model train vs validation loss and accuracy')
plt.ylabel('loss and acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss', 'train acc', 'validation acc'], loc='upper right')
plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Or generate predictions on new data:
#classes = bt_model.predict(X_test, batch_size=128)

### TODO try predict....







#  __     ___                 _ _                            __                                           
#  \ \   / (_)___ _   _  __ _| (_)_______   _ __   ___ _ __ / _| ___  _ __ _ __ ___   __ _ _ __   ___ ___ 
#   \ \ / /| / __| | | |/ _` | | |_  / _ \ | '_ \ / _ \ '__| |_ / _ \| '__| '_ ` _ \ / _` | '_ \ / __/ _ \
#    \ V / | \__ \ |_| | (_| | | |/ /  __/ | |_) |  __/ |  |  _| (_) | |  | | | | | | (_| | | | | (_|  __/
#     \_/  |_|___/\__,_|\__,_|_|_/___\___| | .__/ \___|_|  |_|  \___/|_|  |_| |_| |_|\__,_|_| |_|\___\___|
#                                          |_|                                                            

# # Let's Visualize the model performance by plotting the actual and predicted values. We will plot the values over a small window. It will be quite interesting and new for you guys
# fig, ax1 = plt.subplots(1,1)                                                                        # We will the plot and grab axis
# ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])                   # Create axis for the plot
# ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y') for i in range(2013,2019) for j in [1,5,9]])
# ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime), training_set['bt_Close'][window_len:], label='Actual')  # Let's plot actual data
# ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime), ((np.transpose(bt_model.predict(X_train))+1) *\
#         training_set['bt_Close'].values[:-window_len])[0], label='Predicted')                       # plot predicted data
# ax1.set_title('Training Set: Single Timepoint Prediction')                                          # Set titles for the plot
# ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)                                                     # Set label for y-axis
# ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(bt_model.predict(X_train))+1) - (training_set['bt_Close'].values[window_len:])/\
#             (training_set['bt_Close'].values[:-window_len]))), xy=(0.75, 0.9),  xycoords='axes fraction', xytext=(0.75, 0.9), textcoords='axes fraction') # Plot mean absolute error for defined window size
# ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, 
#            borderaxespad=0., prop={'size': 14})


# # Here is the interesting part. We will create a Zoomed window for a small section of the price history to check how well our model fit on the training data zoom-factor: 2.52, location: centre
# axins = zoomed_inset_axes(ax1, 2.52, loc=10, bbox_to_anchor=(400, 307)) 
# axins.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
# axins.plot(model_data[model_data['Date'] < split_date]['Date'][window_len:].astype(datetime.datetime), training_set['bt_Close'][window_len:], label='Actual')   # Plot Actual data in the zoomed window
# axins.plot(model_data[model_data['Date'] < split_date]['Date'][window_len:].astype(datetime.datetime), ((np.transpose(bt_model.predict(X_train))+1) *\
#            training_set['bt_Close'].values[:-window_len])[0], label='Predicted')                    # Plot Predicted data in the zoomed window
# axins.set_xlim([datetime.date(2017, 2, 15), datetime.date(2017, 5, 1)])                             # Set axis values 
# axins.set_ylim([920, 1400])
# axins.set_xticklabels('')
# mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
# plt.show()


# fig, ax1 = plt.subplots(1,1)                                                                        # Plot results on test data
# ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])                                      # Set axis properties
# ax1.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y') for i in range(12)])
# ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][10:].astype(datetime.datetime), test_set['bt_Close'][window_len:], label='Actual')    # Create plot of actual values for defined time period
# ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][10:].astype(datetime.datetime), ((np.transpose(bt_model.predict(X_test))+1) *\
#            test_set['bt_Close'].values[:-window_len])[0], label='Predicted')                        # Load and test the model on test data and create plot
# ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(bt_model.predict(X_test))+1)-(test_set['bt_Close'].values[window_len:])/(test_set['bt_Close'].values[:-window_len]))),
#              xy=(0.75, 0.9),  xycoords='axes fraction', xytext=(0.75, 0.9), textcoords='axes fraction') # Calculate mean absolute error and plot it
# ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
# ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
# ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
# plt.show()                                                                                          # Plot the results


print("\nDone.")
