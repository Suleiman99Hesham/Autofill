

from collections import defaultdict
import os
import nltk

model = defaultdict(lambda: defaultdict(lambda: 0))

# path = "Sports"
# data_files = os.listdir(path)
# all_data=""
# for i in data_files:
#     if i.endswith(".html"):
#         with open(path+"//"+i,'r',encoding="utf8") as file:
#             for line in file:
#                 all_data+=line
#
# f = open('data_Sports.txt' , 'w' ,  encoding="utf8")
# f.write(all_data)
# f.close()

file = open("data_Sports.txt", "r", encoding="utf8")
data = file.read()
file.close()

data_tokens = nltk.word_tokenize(data)
# print(data_tokenized)

# data_trigram = nltk.trigrams(data_tokens)
# print(list(trigram_data)[-1:])
data_trigram=[]
for i in range(len(data_tokens)-2):
    data_trigram.append((data_tokens[i],data_tokens[i+1],data_tokens[i+2]))


for words in data_trigram:
    w1, w2, w3 = words
    # print("w1:",w1,"\n","w2:",w2,"\n","w3:",w3,"\n")
    model[(w1, w2)][w3] += 1
# print(model[('هذا', 'التخبط')])

count = 0
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    # print(w1_w2,"\t\t",model[w1_w2].values())
    # print(max(model[w1_w2].items(), key=lambda a: a[1]))
    # print(total_count)
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
    # if w1_w2==('الإدارة', 'العامة'):
    #     print(max(model[w1_w2].items(), key=lambda a:a[1]))

# print(model[('الإدارة', 'العامة')].values())
#test model
sentence=input("type: ")
for counter in range(10):
    sentence_temp = nltk.word_tokenize(sentence)
    sentence_temp = tuple(sentence_temp[-2:])
    # print(sentence_temp)
    for w1_w2 in model:
        if sentence_temp == w1_w2:
            max_prop_word = max(model[w1_w2].items(), key=lambda a: a[1])
            sentence += " " + max_prop_word[0]
print(sentence)