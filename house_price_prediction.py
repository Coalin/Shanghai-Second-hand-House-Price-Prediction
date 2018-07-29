#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Date: June 19, 2018
Author: Zhou Jing
Address: colinfdu@gmail.com

Please run this code under the following environment:
1. python 2.7
2. tensorflow 1.1.0
"""

from tkinter import *
import numpy as np
from pandas.core.frame import DataFrame
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import pickle


class Page(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)

    def show(self):
        self.lift()


class Page1(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        # ==============
        # all StringVar()
        # ==============
        Location = StringVar()
        Area = StringVar()
        Towards = StringVar()
        Height = StringVar()
        Decoration = StringVar()
        Age = StringVar()
        lat = StringVar()
        lng = StringVar()
        # Mitoses = StringVar()

        Location.set("")
        Area.set("")
        Towards.set("")
        Height.set("")
        Decoration.set("")
        Age.set("")
        lat.set("")
        lng.set("")

        # =============
        # main layout
        # =============
        Tops = Frame(self, width=1366, height=40, bd=1, relief=RIDGE)
        Tops.pack(side=TOP)

        lblInfo = Label(Tops, font=('arial', 20, 'bold'), bg='purple', fg='white',
                        text="Shanghai Second-Hand House Price Prediction",
                        bd=5, anchor='w')
        lblInfo.grid(row=0, column=0)

        fs1 = Frame(self, width=150, height=0, bd=1, relief='flat')
        fs1.pack(side=LEFT)

        f1 = Frame(self, width=600, height=600, bd=8, relief='raise')
        f1.pack(side=LEFT)

        f2 = Frame(self, width=366, height=600, bd=8, relief='raise')
        f2.pack(side=LEFT)

        # for description
        f3 = Frame(f2, width=1000, height=320, bd=8, relief=FLAT)
        f3.pack(side=BOTTOM)

        f2ab = Frame(f2, width=366, height=300, bd=8, relief='raise')
        f2ab.pack(side=BOTTOM)

        f2a = Frame(f2, width=366, height=250, bd=8, relief='raise')
        f2a.pack(side=BOTTOM)

        f2aa = Frame(f2, width=366, height=250, bd=8, relief='raise')
        f2aa.pack(side=BOTTOM)

        # ===============
        # some displays
        # ===============
        text = 'NOTE:\nTraining Under the Data of April 2018\n' \
               'With the Neural Network(2 Full Connected 200-Layers and 25% Dropout)\n' \
               'Epoch: 10000; MSE: 0.0368985943496\n' \
               'For Example, Location: Yangpu; Area: 75; Towards: N; Height: L; Decoration: 3;\n' \
               'Age: 2001; Lat: 31.307052; Lng: 121.503814'
        lbldetails = Label(f3, font=('Courier', 12), text=text, bd=16, justify='left')
        lbldetails.grid(row=0, column=0)

        # ============
        #  Inputs
        # ============
        lblLocation = Label(f1, font=('arial', 16, 'bold'), text='Location', bd=16, justify='left')
        lblLocation.grid(row=0, column=0)
        txtLocation = Entry(f1, font=('arial', 16, 'bold'), textvariable=Location, bd=10, insertwidth=2,
                                  justify='left')
        txtLocation.grid(row=0, column=1)

        lblArea = Label(f1, font=('arial', 16, 'bold'), text='Area', bd=16,
                                        justify='left')
        lblArea.grid(row=1, column=0)
        txtArea = Entry(f1, font=('arial', 16, 'bold'), textvariable=Area, bd=10,
                                        insertwidth=2, justify='left')
        txtArea.grid(row=1, column=1)

        lblTowards = Label(f1, font=('arial', 16, 'bold'), text='Towards', bd=16,
                                         justify='left')
        lblTowards.grid(row=2, column=0)
        txtTowards = Entry(f1, font=('arial', 16, 'bold'), textvariable=Towards, bd=10,
                                         insertwidth=2, justify='left')
        txtTowards.grid(row=2, column=1)

        lblHeight = Label(f1, font=('arial', 16, 'bold'), text='Height', bd=16, justify='left')
        lblHeight.grid(row=3, column=0)
        txtHeight = Entry(f1, font=('arial', 16, 'bold'), textvariable=Height, bd=10, insertwidth=2,
                                    justify='left')
        txtHeight.grid(row=3, column=1)

        lblDecoration = Label(f1, font=('arial', 16, 'bold'), text='Decoration', bd=16,
                                            justify='left')
        lblDecoration.grid(row=4, column=0)
        txtDecoration = Entry(f1, font=('arial', 16, 'bold'), textvariable=Decoration,
                                            bd=10, insertwidth=2, justify='left')
        txtDecoration.grid(row=4, column=1)

        lblAge = Label(f1, font=('arial', 16, 'bold'), text='Age', bd=16, justify='left')
        lblAge.grid(row=5, column=0)
        txtAge = Entry(f1, font=('arial', 16, 'bold'), textvariable=Age, bd=10, insertwidth=2,
                              justify='left')
        txtAge.grid(row=5, column=1)

        lbllat = Label(f1, font=('arial', 16, 'bold'), text='Latitude', bd=16, justify='left')
        lbllat.grid(row=6, column=0)
        txtlat = Entry(f1, font=('arial', 16, 'bold'), textvariable=lat, bd=10, insertwidth=2,
                                  justify='left')
        txtlat.grid(row=6, column=1)

        lbllng = Label(f1, font=('arial', 16, 'bold'), text='Longitude', bd=16, justify='left')
        lbllng.grid(row=7, column=0)
        txtlng = Entry(f1, font=('arial', 16, 'bold'), textvariable=lng, bd=10, insertwidth=2,
                                  justify='left')
        txtlng.grid(row=7, column=1)

        # ================
        #    Outputs
        # ================
        Result = StringVar()
        lblResult = Label(f2a, font=('arial', 16, 'bold'), text='Average & Overall Price', bd=16, anchor='w')
        lblResult.grid(row=0, column=0)
        txtResult = Entry(f2a, font=('arial', 16, 'bold'), textvariable=Result, bd=10, insertwidth=2, justify='left')
        txtResult.grid(row=0, column=1)

        def getData():
            alist = []
            Location_data = str(Location.get())
            alist.append(Location_data)
            Area_data = int(Area.get())
            alist.append(Area_data)
            Towards_data = str(Towards.get())
            alist.append(Towards_data)
            Height_data = str(Height.get())
            alist.append(Height_data)
            Decoration_data = int(Decoration.get())
            alist.append(Decoration_data)
            Age_data = int(Age.get())
            alist.append(Age_data)
            lat_data = float(lat.get())
            alist.append(lat_data)
            lng_data = float(lng.get())
            alist.append(lng_data)
            return alist

        def normalize(data_frame_encoded):
            data = data_frame_encoded
            data = [np.log(tt + 1) for tt in data]
            return data

        def batch_generator(data):
            num_features = len(data)
            num_batches = len(data[0])
            for i in range(num_batches):
                batch_compiled = []
                for j in range(num_features):
                    if type(data[j][i]) is np.ndarray:
                        batch_compiled.extend(data[j][i])
                    else:
                        batch_compiled.extend([data[j][i]])
                yield batch_compiled

        def oh_encode(data_frame, encoders_path=None):
            data_encoded = []
            encoders = [] if not encoders_path else pickle.load(open(encoders_path, 'rb'))

            if encoders:
                for feature, encoder in zip(data_frame, encoders):
                    data_i = list(data_frame[feature])
                    if encoder is not None:
                        data_i = encoder.transform(data_i)
                    try:
                        data_i = np.array(data_i, dtype=np.float32)
                    except ValueError:
                        for n, i in enumerate(data_i):
                            if i == 'NA':
                                data_i[n] = 0
                        data_i = np.array(data_i, dtype=np.float32)
                    data_encoded.append(data_i)
            else:
                for feature in data_frame:
                    data_i = data_frame[feature]
                    encoder = None
                    if data_frame[feature].dtype == 'O':  # is data categorical?
                        encoder = LabelBinarizer()
                        encoder.fit(list(set(data_frame[feature])))
                        data_i = encoder.transform(data_i)
                    data_i = np.array(data_i, dtype=np.float32)
                    data_encoded.append(data_i)
                    encoders.append(encoder)
                pickle.dump(encoders, open('encoders.pickle', 'wb'))
            return data_encoded

        def printResult():
            try:
                x_in = getData()

            except ValueError:
                Result.set('Insufficient Data!')

            else:
                c = {"Location": x_in[0], "Area": x_in[1],
                     "Towards": x_in[2], "Height": x_in[3],
                     "Decoration": x_in[4], "Age": x_in[5],
                     "lat": x_in[6], "lng": x_in[7]}
                df_train = DataFrame(c, index=[0])
                # df_train['Location'].dtypes = object
                # df_train['Height'].dtypes = object
                # df_train['Towards'].dtypes = object
                df_train = df_train[['Location', 'Area', 'Towards', 'Height', 'Decoration', 'Age', 'lat', 'lng']]
                df_train.to_csv("result.csv")

                # df_train = pd.read_csv('/Users/Colin/PycharmProjects/house_price_prediction/data/new_test.csv')
                # df_train = df_train.drop(['Id'], 1)
                df_train_encoded = oh_encode(df_train, encoders_path='encoders.pickle')
                df_train_encoded_normalized = normalize(df_train_encoded)
                vars = pickle.load(open('./saves/weights.pickle', 'rb'))
                input_layer = tf.placeholder(tf.float32, [None, 626])
                W1 = tf.constant(vars[0])
                b1 = tf.constant(vars[1])
                h1_layer = tf.add(tf.matmul(input_layer, W1), b1)
                h1_layer = tf.nn.relu(h1_layer)
                h1_layer = tf.nn.dropout(h1_layer, 1)

                W2 = tf.constant(vars[2])
                b2 = tf.constant(vars[3])
                h2_layer = tf.matmul(h1_layer, W2) + b2
                h2_layer = tf.nn.relu(h2_layer)
                h2_layer = tf.nn.dropout(h2_layer, 1)

                W3 = tf.constant(vars[4])
                b3 = tf.constant(vars[5])
                output_layer = tf.matmul(h2_layer, W3) + b3

                gen = batch_generator(df_train_encoded_normalized)

                all_batches = [b for b in gen]

                prices = None

                with tf.Session() as sess:
                    prices = sess.run([output_layer], feed_dict={input_layer: all_batches})
                    prices = np.array(prices).flatten()
                    prices = np.exp(prices) + 1.
                    print(prices)
                    sess.close()

                result = round(prices[0], 2)
                overall = round(result * df_train['Area'], 2)
                if result:
                    Result.set(str(result) + ' 元/平方米; ' + str(overall) + ' 元')
                else:
                    Result.set('ValueError!')

        def Reset():
            Location.set("")
            Area.set("")
            Towards.set("")
            Height.set("")
            Decoration.set("")
            Age.set("")
            lat.set("")
            lng.set("")
            # Mitoses.set("")
            Result.set("")

        btnTotal = Button(f2aa, padx=16, pady=16, bd=8, fg='black', font=('arial', 16, 'bold'), width=15,
                          text='Predict', command=printResult).grid(row=0, column=0)
        btnReset = Button(f2ab, padx=16, pady=16, bd=8, fg='black', font=('arial', 16, 'bold'), width=15, text='Reset',
                          command=Reset).grid(row=0, column=0)


class MainView(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)

        buttonframe = Frame(self)
        container = Frame(self)
        buttonframe.pack(side=TOP, fill=X, expand=False)
        container.pack(side=TOP, fill='both', expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = Button(buttonframe, text="Shanghai Second-Hand House Price Prediction", command=p1.lift)

        b1.pack(side=LEFT)

        p1.show()


if __name__ == "__main__":
    root = Tk()
    main = MainView(root)
    root.title("Shanghai Second-Hand House Prediction Tools")
    main.pack(side=TOP, fill='both', expand=True)
    root.wm_geometry('1366x768')
    root.mainloop()
