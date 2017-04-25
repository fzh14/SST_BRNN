sentence = 'Effective but too-tepid biopic'
# dict of 'sentences'---'id'
fp = open('dictionary.txt', 'r')
dict = {}
count = 0
for i in fp.readlines():
    count += 1
    line = i.strip('\n').split('|')
    if len(line) != 2:
        print line
    dict[line[0]] = line[1]

fp.close()

# dict2 of 'id'---labels
dict2 = {}
f2 = open('sentiment_labels.txt', 'r')
for i in f2.readlines():
    line = i.strip('\n').split('|')
    if len(line) == 2:
        dict2[line[0]] = line[1]

f2.close()

# write output file
fp = open('datasetSentences.txt', 'r')
output = open('output_dataset.txt', 'w')
count = 0
error = 0
for i in fp.readlines():
    count += 1
    if count == 1:
        pass
    else:
        arr = i.strip('\n').split('\t')
        try:
            index = dict[arr[1]]
            s = ''
            s = arr[1]
            s += '|'
            label = dict2[index]
            s += label +'|'+ arr[0] +'\n'
            output.write(s)
        except:
            error += 1

print error
