import preprocess as read
import matplotlib.pyplot as plt
import numpy as np



#data = read.read_from_dir("valList.txt")

# for line in data.splitlines():
#     print line, int(result_dict[line][0]),int(result_dict[line][1]),float(result_dict[line][2])

# count = 0
# for key,value in result_dict.items():
#     if result_dict[key][0] == "0":
#         count+=1
# print count


positive_count = 16
negative_count = 70


precision = list()
recall = list()
accuracy = list()
f1m = list()
index = []

training_result = dict()
print "Person","True label","Prediction","Probability"
for folder in range(5):
    exp1 = "fold_regex_metamap/model_regex_metamap"
    epoch = "78_fold" + str(folder)
    exp = exp1 + "/" + epoch
    result_dict = read.read_from_json(exp + "/result_dict")

    for key,value in result_dict.items():
        training_result[key] = value
        print key,value[0],value[1],value[2]
read.save_in_json(exp1+"/training_result",training_result)

# folder =4
# exp1 = "fold_regex_metamap/model_regex_metamap"
# epoch = "78_fold" + str(folder)
# exp = exp1 + "/" + epoch
# training_result =read.read_from_json(exp + "/result_dict")
# for i in np.linspace(0,1,num=100):
for i in [0.5]:
    # i=float(i)
    print i
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for key, value in training_result.items():
        if float(training_result[key][2]) >= i and training_result[key][0] == "1":
            tp+=1.0
        elif float(training_result[key][2]) < i and training_result[key][0] == "0":
            tn+=1.0
        elif float(training_result[key][2]) >= i and training_result[key][0] == "0":
            fp +=1.0
        elif float(training_result[key][2]) < i and training_result[key][0] == "1":
            fn += 1.0
    if tp+fp == 0.0:
        pre = 0.0
    else:
        pre = tp/(tp+fp)
    if tp+fn == 0.0:
        rec=0.0
    else:
        rec = tp/(tp+fn)
    acc = (tp+tn)/86.0
    if pre+rec ==0:
        f1 = 0.0
    else:
        f1 = 2*(pre*rec)/(pre+rec)
    index.append(i)
    precision.append(pre)
    recall.append(rec)
    accuracy.append(acc)
    f1m.append(f1)
print precision
print recall
print f1m






# plt.plot(index, precision)
# plt.plot(index, recall)
# plt.plot(index, accuracy)
# plt.plot(index, f1m)
# #plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
# plt.legend(['precision', 'recall', 'accuracy', 'F1 score'], loc='best')
# plt.xlabel('Threshold')
# plt.ylabel('Scores')
# #plt.title('acc: 0.8953; f1: 0.7273; pre: 0.7059; rec: 0.75. (regex & metamap features)')
# #plt.title('acc: 0.8837; f1: 0.7050; pre: 0.6666; rec: 0.75. (regex features)')
# plt.show()

