import os
import json
import h5py
import numpy as np
import cPickle


def save_in_cPickle(filename, array):
    cPickle.dump(array, open(filename, "wb"))

def read_from_cPickle(filename):
    array = cPickle.load(open(filename, "rb"))
    return array



def save_in_json(filename, array):
    with open(filename, 'w') as outfile:
        json.dump(array, outfile)
    outfile.close()

def read_from_json(filename):
    with open(filename, 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data

def read_from_dir(path):
    data =open(path).read()
    return data

def textfile2list(path):
    data = read_from_dir(path)
    list_new =list()
    for line in data.splitlines():
        list_new.append(line)
    return list_new

def load_input(filename):
    """

    :param filename: input file name
    :return:
    x_data: 3-ds array,1-d user,2-d post, 3-d feature vector
    y_data: 2-ds array, 1-d user,2-d label
    """

    with h5py.File( filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')
        y = hf.get('output')
        x_data = np.array(x)
        y_data = np.array(y)
        print x_data.shape
        print y_data.shape
    return x_data,y_data

def load_test(filename):
    """

    :param filename: test input file name
    :return:
    x_data: 3-ds array,1-d user,2-d post, 3-d feature vector
    """
    with h5py.File(filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')
        x_data = np.array(x)
        # n_patterns = x_data.shape[0]
        #y_data = np.array(y)
        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        # y_data = y_data.reshape(y_data.shape+(1,))
        print x_data.shape
        #print y_data.shape
    return x_data

def get_rawdata_dir():
    '''
    get the directory for the whole raw data, return two lists, one list for the positive users' file names, the other for negative users' file names
    :return: positive_folder,negative_folder
    '''
    dirname = "data/"

    roots = os.listdir(dirname)
    positive_folder = list()
    negative_folder =list()
    for root in roots:
        root_com =os.path.join(dirname,root)
        if 'positive' in root:
            positive_folder += [os.path.join(root_com,f) for f in os.listdir(root_com) if not os.path.isdir(os.path.join(root_com, f))]
        else:
            negative_folder += [os.path.join(root_com,f) for f in os.listdir(root_com) if not os.path.isdir(os.path.join(root_com, f))]
    return positive_folder,negative_folder

def goldtruth2dict ():
    """
    risk_golden_truth file: each line is one instance, such as: "train_subject669 1"
    Read from risk_golden_truth.txt file, return a dictionary where key is the file name, and the value is the label
    :return:
    """
    gold_dict =dict()
    gold = read_from_dir("risk_golden_truth")
    for line in gold.splitlines():
        text = line.split(" ")
        gold_dict[text[0]] = text[1]
    #print gold_dictu
    return gold_dict

# gold_dict = goldtruth2dict()
# save_in_json('gold_dict',gold_dict)

def getlistwithlabel(labels,gold_dict):
    list_dict = dict()
    for label in labels:
        if gold_dict[label] =='1':
            list_dict[label] ='1'
        else:
            list_dict[label] = '0'
    return list_dict

def datawithlabel():
    """
    read from trainList.txt and valList.txt, split the gold_dict into train_dict and val_dict and save into json files.

    """
    train = textfile2list('data/trainList.txt')
    val = textfile2list('data/valList.txt')
    gold_dict = goldtruth2dict()
    train_dict = getlistwithlabel(train,gold_dict) #### split the gold_dict into train_dict and val_dict
    val_dict = getlistwithlabel(val,gold_dict)
    save_in_json('data/train_dict',train_dict)
    save_in_json('data/val_dict', val_dict)

# datawithlabel()

def get_features(features,features_for,features_folder,separator):
    """
    used for regex, pmiWords, metamap, processed features_folder into a dictionary, key: user_name "train_subject3670", value: 2-d array post-level features
    :param features: string type, features name
    :param features_for: val or train
    :param features_folder: processed feature vector's folder, such as "pmiWordsOutput","metamapOutput"
    :param separator: separator used to separate the items of feature vector, "," or " " or ":"
    """
    train_dict = read_from_json(features_for+"_dict")
    currdir = os.getcwd()
    os.chdir('%s/' % features_folder)
    post_count = list()
    user = dict()
    for key,value in train_dict.items():
        print key
        data = read_from_dir(key+".txt")
        post = list()
        for line in data.splitlines():
            text = line.split(separator)
            text.pop(0)
            post.append(map(int,text))
        post_count.append(len(post))
        user[key] =post
    os.chdir(currdir)

    save_in_json(features+"_"+features_for +"_dict", user)  ######
    save_in_json(features+"_"+features_for +"_post_count", post_count)
    print max(post_count)      #### the maximal count of the posts among all users
    print post_count.index(2000)
    print min(post_count)

#get_features("features_all/regex","train","features_all/regexOutput",":")
#get_features("features_all/regex","val","features_all/regexOutput",":")
#get_features("features_all/metamap","train","features_all/metamapOutput",",")
#get_features("features_all/metamap","val","features_all/metamapOutput",",")



def get_post_embedding(features,features_for,features_folder,separator):
    """
    used for embedding, processed features_folder into a dictionary, key: user_name "train_subject3670", value: 2-d array post-level features
:param features: string type, features name
    :param features_for: val or train
    :param features_folder: processed feature vector's folder, such as "pmiWordsOutput","metamapOutput"
    :param separator: separator used to separate the items of feature vector, "," or " " or ":"
    """
    train_dict = read_from_json(features_for + "_dict")
    currdir = os.getcwd()
    os.chdir('%s/' % features_folder)
    post_count = list()
    user = dict()
    for key, value in train_dict.items():
        print key
        data = read_from_dir(key + ".txt")
        post = list()
        for line in data.splitlines():
            text = line.split(separator)
            text.pop(64)
            post.append(map(np.float32,text))
        post_count.append(len(post))
        user[key] = post
    os.chdir(currdir)

    save_in_cPickle(features+"_"+features_for + "_dict", user)  ######
    save_in_json(features +"_"+features_for + "_post_count", post_count)
    print max(post_count)  #### the maximal count of the posts among all users
    print post_count.index(2000)
    print min(post_count)

#get_post_embedding("features_all/embedding","train","features_all/embedding"," ")
#get_post_embedding("features_all/embedding","val","features_all/embedding"," ")


def concatenate_features(user_postfeatures,feature_size,train_dict, outputfilename):

    data_size = len(train_dict)
    max_post_size = 2000


    f = h5py.File(outputfilename + ".hdf5", "w")
    dset = f.create_dataset("input", (data_size, max_post_size, feature_size), dtype='float32')
    dset2 = f.create_dataset("output", (data_size, 1), dtype='float32')  #### commment this one for get test_data
    iter = 0
    for key,value in train_dict.items():
        print key, train_dict[key]
        #if gold_case[0] == "train_subject3615":
        #     print "asdawd"



        new_post = list()
        user_postfeature0 = user_postfeatures[0]
        n_post = len(user_postfeature0[key])

        for j in range(n_post):
            post_feature = []
            for user_postfeature in user_postfeatures:
                post_feature+=user_postfeature[key][j]
            new_post.append(post_feature)
            #new_post.append(posts[j] + cuis[j])
        dset[iter, :n_post, :] = new_post
        # for i in range(gap,data_size):
        dset[iter, n_post:, :] = np.float32(-1.0)
        dset2[iter, 0] = [np.float32([1])]
        iter += 1


def get_rnn_input(feature_lists,features_for,feature_size,input_name):
    """

    :param feature_lists: a list of features, such as ["pmi_word",]
    :return:
    """
    train_dict = read_from_json(features_for + "_dict")

    user_postfeatures = list()
    for feature in feature_lists:
        if not feature == "embedding":
            user_postfeatures.append(read_from_json("features_all/"+feature +"_"+features_for + "_dict"))
        else:
            user_postfeatures.append(read_from_cPickle("features_all/"+feature +"_"+features_for + "_dict"))

    concatenate_features(user_postfeatures, feature_size, train_dict, input_name)

get_rnn_input(["regex","metamap","embedding"],"val",110+404+64,"features_all/val_regex_matamap_embedding")









# get_post_embedding("positive_train"," ")    #67
# get_post_embedding("negative_val"," ")       #70
# get_post_embedding("negative_train"," ")     #333

#k = read_from_pickle("post_level_embedding/positive_val_dict")

# def get_post_count_new(path,name):
#     train_dict = read_from_json(path)
#     currdir = os.getcwd()
#     path = "regexOutput"
#     os.chdir('%s/' % path)
#     post_count = list()
#     user = dict()
#     for key,value in train_dict.items():
#         print key
#         data = read_from_dir(key+".txt")
#         post = list()
#         for line in data.splitlines():
#             text = line.split(":")
#             text.pop(0)
#             text = map(int,text)
#             index = 1
#             for feature in text:
#                 if feature !=0:
#                     for k in range(feature):
#                         post.append(index)
#                 index +=1
#         post_count.append(len(post))
#         user[key] =post
#     os.chdir(currdir)
#     save_in_json(name,user)
#     save_in_json("post_count_v2", post_count)
#     print max(post_count)
#     print post_count.index(2654)
#     print min(post_count)

# get_cui_count('val_dict','users_val_cui')
#get_cui_count('train_dict','users_train_cui')



# user_dict = read_from_json('train_dict')
# for key,value in user_dict.items():
#     print key,value

# get_post_count_new('train_dict','users_wtrain_v2')

#get_post_count_new('val_dict','users_val_v2')
# users_train = read_from_json('users_val_v2')
# post = users_train['train_subject5967']
# print len(post)





# def get_training(users_train,label_dict,outputfilename):
#     data_size = len(users_train)
#     max_post_size = 2000
#     feature_size = 2000
#     f = h5py.File(outputfilename+".hdf5", "w")
#     dset = f.create_dataset("input", (data_size,max_post_size,feature_size), dtype='int8')
#     dset2 = f.create_dataset("output", (data_size,1), dtype='int8')
#     iter = 0
#     for key,label in label_dict.items():
#         print key
#         print label
#         posts = users_train[key]
#         gap =len(posts)
#         dset[iter,:gap,:] = posts
#         # for i in range(gap,data_size):
#         dset[iter,gap:,:] = -1.0
#         dset2[iter,0] =[int(label)]
#         iter+=1


# users_train = read_from_json('pmi_word_val_dict')
# train_dict = read_from_json('val_dict')
# get_training(users_train,train_dict,'val_data_pmi')

# def get_training_cui_regex(users_train,user_train_cui,label_dict, outputfilename):
#     data_size = len(users_train)
#     max_post_size = 2000
#     feature_size = 110+404
#     f = h5py.File(outputfilename + ".hdf5", "w")
#     dset = f.create_dataset("input", (data_size, max_post_size, feature_size), dtype='int8')
#     dset2 = f.create_dataset("output", (data_size, 1), dtype='int8')
#     iter = 0
#     for key, label in label_dict.items():
#         print key
#         print label
#         posts = users_train[key]
#         cuis = user_train_cui[key]
#         new_post = list()
#         gap = len(posts)
#         for j in range(gap):
#             new_post.append(posts[j]+cuis[j])
#         dset[iter, :gap, :] = new_post
#         # for i in range(gap,data_size):
#         dset[iter, gap:, :] = -1.0
#         dset2[iter, 0] = [int(label)]
#         iter += 1
#
# def get_training_post_embedding(data_size,positive_list,negative_list,label_dict_positive,label_dict_negative, outputfilename):
#
#     max_post_size = 2000
#     feature_size = 64
#     f = h5py.File(outputfilename + ".hdf5", "w")
#     dset = f.create_dataset("input", (data_size, max_post_size, feature_size), dtype='float32')
#     dset2 = f.create_dataset("output", (data_size, 1), dtype='float32')
#     iter = 0
#     for file in positive_list:
#         print file
#         posts = label_dict_positive[file]
#
#         new_post = list()
#         gap = len(posts)
#         for j in range(gap):
#             new_post.append(posts[j])
#         dset[iter, :gap, :] = new_post
#         # for i in range(gap,data_size):
#         dset[iter, gap:, :] = -1.0
#         dset2[iter, 0] = [1]
#         iter += 1
#     for file in negative_list:
#         print file
#         posts = label_dict_negative[file]
#
#         new_post = list()
#         gap = len(posts)
#         for j in range(gap):
#             new_post.append(posts[j])
#         dset[iter, :gap, :] = new_post
#         # for i in range(gap,data_size):
#         dset[iter, gap:, :] = -1.0
#         dset2[iter, 0] = [0]
#         iter += 1

# get_post_embedding("positive_val"," ")    #16
# get_post_embedding("positive_train"," ")    #67
# get_post_embedding("negative_val"," ")       #70
# get_post_embedding("negative_train"," ")     #333

# dir = "post_level_embedding/"
# train = "val"
# data_size= 86   #400
# feature = "positive_"+train
# feature1 = "negative_"+train
# positive_list = read_from_json(dir+feature+"_list")
# negative_list = read_from_json(dir+feature1+"_list")
#
# label_dict_positive = read_from_pickle(dir+feature+"_dict")
# label_dict_negative = read_from_pickle(dir+feature1 + "_dict")
#
# get_training_post_embedding(data_size,positive_list,negative_list,label_dict_positive,label_dict_negative, dir+train+"_pos_embedding")



# x_data,y_data = load_input('post_level_embedding/val_pos_embedding')
# print x_data.shape
# print x_data[50][1083]








# chunk ="chunk10"
# dir = "prediction/"+chunk+"/"
# users_train = read_from_json(dir+'regexOutputTest_dict')
# user_train_cui = read_from_json(dir+'metamapFeatureVectorsTest_dict')
# roots = read_from_json(dir+'roots')
#
# get_test_data(users_train, user_train_cui, roots, dir+"test_predict_regex_"+chunk, use_cui=False)
# get_test_data(users_train, user_train_cui, roots, dir+"test_predict_regex_metamap_"+chunk, use_cui=True)



# x_data = load_test(dir+'test_predict_regex_'+chunk)
# x1_data = load_test(dir+'test_predict_regex_metamap_'+chunk)
# print x_data[0][57]
# print x1_data[0][57]





# def get_training_count_new(users_train, label_dict, outputfilename):
#     data_size = len(users_train)
#     max_post_size = 3596
#     feature_size = 110
#     f = h5py.File(outputfilename + ".hdf5", "w")
#     dset = f.create_dataset("input", (data_size, max_post_size), dtype='int8')
#     dset2 = f.create_dataset("output", (data_size, 1), dtype='int8')
#     iter = 0
#     for key, label in label_dict.items():
#         print key
#         print label
#         posts = users_train[key]
#         print posts
#         data_x = pad_sequences([posts], dtype='int8', maxlen=max_post_size, padding="pre")
#         dset[iter, :] = data_x[0]
#         dset2[iter, 0] = [int(label)]
#         iter += 1

# users_train = read_from_json('users_train')
# user_train_cui = read_from_json('users_train_cui')
# label_dict = read_from_json('train_dict')
# get_training_cui_regex(users_train,user_train_cui,label_dict,'training_data_merged')

# users_train = read_from_json('users_val')
# user_train_cui = read_from_json('users_val_cui')
# label_dict = read_from_json('val_dict')
# get_training_cui_regex(users_train,user_train_cui,label_dict,'val_data_merged')

# users_train = read_from_json('users_val')
# label_dict = read_from_json('val_dict')
# get_training(users_train,label_dict,'val_data_minusone')

# users_train = read_from_json('users_train_v2')
# label_dict = read_from_json('train_dict')
# get_training_count_new(users_train,label_dict,'training_data_v1')

# users_train = read_from_json('users_val_v2')
# label_dict = read_from_json('val_dict')
# get_training_count_new(users_train,label_dict,'val_data_v1')

#


#
# x_data,y_data = load_input('val_data_merged')
# print x_data.shape
# print x_data[3][209]

# x_data,y_data = load_input('training_data_merged')
# print x_data.shape
# print y_data[20:30]

# x_data,y_data = load_input('training_data')
# # print x_data[100]

#x_data,y_data = load_input('val_data')
# print x_data[100]


# def get_test_dir(data_name,sourcedir,target):
#     '''
#     get the directory for whole raw data and xml data, using the same root dir raw_text_dir
#     :param raw_text_dir: root directory
#     :return: both raw_data directory and xml_data directory
#     '''
#     # currdir = os.getcwd()
#     #dirname = "prediction/chunk1"
#
#     # os.chdir('%s/' % sourcedir)
#     roots = os.listdir(sourcedir+ data_name+"/")
#     save_in_json(target+data_name+"_list",roots)
#     # os.chdir(currdir)

#get_prediction_dir("negative_train","post_level_embedding/test_vectors_","post_level_embedding/")

# def get_test_regex(features,prefix):
#     currdir = os.getcwd()
#     dirname = "prediction/chunk10"
#     os.chdir('%s/' % dirname)
#     roots = read_from_json("roots")
#     print len(roots)
#     post_count = list()
#     user = dict()
#     for file in roots:
#         print file
#         data = read_from_dir(features+"/"+file)
#         post = list()
#         for line in data.splitlines():
#             text = line.split(prefix)
#             text.pop(0)
#             post.append(map(int,text))
#         post_count.append(len(post))
#         user[file] =post
#     save_in_json(features+"_dict", user)
#     save_in_json(features+"_post_count", post_count)
#     os.chdir(currdir)

# get_prediction_regex("metamapFeatureVectorsTest",",")
# get_prediction_regex("regexOutputTest",":")

# def get_test_data(users_train, user_train_cui, roots, outputfilename,use_cui=False):
#     data_size = len(users_train)
#     max_post_size = 2000
#
#     #dset2 = f.create_dataset("output", (data_size, 1), dtype='int8')
#     iter = 0
#     if use_cui == False:
#         feature_size = 110
#         f = h5py.File(outputfilename + ".hdf5", "w")
#         dset = f.create_dataset("input", (data_size, max_post_size, feature_size), dtype='int8')
#         for file in roots:
#             print file
#             posts = users_train[file]
#             gap = len(posts)
#             dset[iter, :gap, :] = posts
#             # for i in range(gap,data_size):
#             dset[iter, gap:, :] = -1.0
#             iter += 1
#     else:
#         feature_size = 110+404
#         f = h5py.File(outputfilename + ".hdf5", "w")
#         dset = f.create_dataset("input", (data_size, max_post_size, feature_size), dtype='int8')
#         for file in roots:
#             print file
#             posts = users_train[file]
#             cuis = user_train_cui[file]
#             new_post = list()
#             gap = len(posts)
#             for j in range(gap):
#                 new_post.append(posts[j] + cuis[j])
#             dset[iter, :gap, :] = new_post
#             # for i in range(gap,data_size):
#             dset[iter, gap:, :] = -1.0
#             iter += 1