import gensim
import string
import os

fp = open('datasetSplit.txt', 'r')
num_train = 0
num_test = 0
num_dev = 0
count = 0
dict_split = {}
for i in fp.readlines():
    count += 1
    if count == 1:
        pass
    else:
        li = i.strip('\n').split(',')
        dict_split[li[0]] = li[1]

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
model = gensim.models.KeyedVectors.load_word2vec_format(project_path+'/glove.6B/glove.6B.50d.txt',
                                                        binary=False)
num = 0
fd = open('output_dataset.txt', 'r')
f_train = open('train_input.txt', 'w')
f_dev = open('dev_input.txt', 'w')
f_test = open('test_input.txt', 'w')

for i in fd.readlines():
    li = i.strip('\n').split('|')
    word_list = string.lower(li[0]).split(' ')
    try:
        for word in word_list:
            s = model[word].tolist()
        num += 1
        line = li[0] + '|' + li[1] + '\n'
        if dict_split[li[2]] == '1':
            num_train += 1
            f_train.write(line)
        elif dict_split[li[2]] == '2':
            num_test += 1
            f_test.write(line)
        elif dict_split[li[2]] == '3':
            num_dev += 1
            f_dev.write(line)
    except:
        pass

print num_train
print num_test
print num_dev
print num
