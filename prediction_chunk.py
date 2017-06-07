import h5py
import numpy as np
import os
import json
from keras.models import load_model
import csv
from collections import OrderedDict



def load_input(filename):
    with h5py.File( filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')
        y = hf.get('output')
        x_data = np.array(x)
        #n_patterns = x_data.shape[0]
        y_data = np.array(y)
        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        #y_data = y_data.reshape(y_data.shape+(1,))
        # print(x_data.shape)
        # print(y_data.shape)


    del x
    del y
    return x_data,y_data

def read_from_json(filename):
    with open(filename, 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data


def prediction_chunks_postbypost(start,end):
    """

    :param start:  initial post location
    :param end: post stop location
    :return:
    """


    test_list = read_from_json("test_filename_list")  ### a list of test user id
    csv_writer = open("csv_log_file_path" + ".csv", 'w')   ### path to save csv logfile


    data_x,data_y = load_input("test_data")  #### the path of your test data
    exp = "best_model"   #### the path of your  best model
    model = load_model(exp + ".hdf5")

    writer = csv.DictWriter(csv_writer,
                            fieldnames=['chunk'] + test_list)

    for chunk in range(start,end):
        prob_list = list()

        val_x = data_x.copy()
        val_x[:, chunk:, :] = -1
        y_prob = model.predict_proba(data_x)
        y_prob = np.asarray(y_prob).flatten()

        prob_list += list(y_prob)
        print y_prob
        row_dict =OrderedDict({"chunk":chunk})
        row_dict.update((test_list[i],prob_list[i]) for i in range(len(prob_list)))
        writer.writerow(row_dict)
        csv_writer.flush()
    csv_writer.close()


# def prediction_chunks_folder(features,feature1,start,end):
#     data_x = list()
#     models = list()
#     val_list = read_from_json(features+feature1+"/val_list")
#     csv_writer = open(features+ feature1+ ".csv", 'w')
#
#
#     writer = csv.DictWriter(csv_writer,
#                             fieldnames=['chunk']+val_list)
#
#     writer.writeheader()
#     for folder in range(2):
#         val_x1, val_y = load_input(features + feature1 + "/fold/" + feature1 + "" + str(folder) + "_val")
#         exp = features + feature1 + "/" + feature1 + str(folder)
#         model1 = load_model(exp + ".hdf5")
#         data_x.append(val_x1)
#         models.append(model1)
#
#
#
#     for chunk in range(start,end):
#         prob_list = list()
#         for folder in range(2):
#             val_x = data_x[folder]
#             val_x[:, chunk:, :] = -1
#             y_prob = models[folder].predict_proba(data_x[folder])
#             y_prob = np.asarray(y_prob).flatten()
#
#             prob_list += list(y_prob)
#             print y_prob
#         row_dict =OrderedDict({"chunk":chunk})
#         row_dict.update((val_list[i],prob_list[i]) for i in range(len(prob_list)))
#         writer.writerow(row_dict)
#         csv_writer.flush()
#     csv_writer.close()
#
# features = "experiment_all/embedding_post/"
# feature1 = "embedding_ngram_metamap"
# prediction_chunks_folder(features,feature1,1,2)