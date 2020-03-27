from collections import defaultdict
import nltk

# creating the model
model = defaultdict(lambda: defaultdict(lambda: 0))

print("Please wait...!")

# reading the data
file = open("Sports_Data.txt", "r", encoding="utf8")
data = file.read()
file.close()

# split data in single words
data=data.split()
data_tokens = [w.strip(" ") for w in data]


# creating trigrams from tokens
data_trigram=[]
for i in range(len(data_tokens)-2):
    data_trigram.append((data_tokens[i],data_tokens[i+1],data_tokens[i+2]))

# rearrange the data to prepare it for calculating probabilities of co-occers
for words in data_trigram:
    w1, w2, w3 = words
    model[(w1, w2)][w3] += 1

# calculating probabilities of co-occers
count = 0
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

# here is the test of model

# sentence="المرحلة التنافسية"  #this for fixed input
sentence=input("write your words: ")  #for dynamic input
print("loading...!")


# option1
# predicting the ten words after getting some words (more than two) / predicting complete sentence

# for counter in range(10):
#     sentence_temp = nltk.word_tokenize(sentence)
#     sentence_temp = tuple(sentence_temp[-2:])
#     for w1_w2 in model:
#         # print(w1_w2)
#         if sentence_temp == w1_w2:
#             max_prop_word = max(model[w1_w2].items(), key=lambda a: a[1])
#             sentence += " " + max_prop_word[0]
#
# # the final result
# print("predicted sentence: ",sentence)


#Option2
#showing a list of predicted words and choose between them

while True:
    sentence_temp = nltk.word_tokenize(sentence)
    sentence_temp = tuple(sentence_temp[-2:])
    for w1_w2 in model:
        if sentence_temp == w1_w2:
            print(sentence,"...")
            i=1
            predictions=sorted(model[w1_w2].items(),key=lambda t:(-t[1],t[0]))
            for predicted_word in predictions:
                if i==7:break
                print(i,"-",predicted_word[0])
                i+=1
            choosed_word=int(input(">>"))
            sentence+=" "+predictions[choosed_word-1][0]
    option = input("continue? Y or N").lower()
    while option!="y" and option!="n":
        print("Oops, wrong answer!")
        option = input("continue? Y or N").lower()
    if option == "y":
        sentence +=" "+input(sentence+" ")
        print("loading...!")
    else :
        break