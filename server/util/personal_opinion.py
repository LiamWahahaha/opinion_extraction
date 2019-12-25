import os
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import defaultdict

currentDirectory = os.getcwd()
#print(currentDirectory)

'''
    spoken_words 来自于我们提前训练好的模型--lang model.加载lang model 从而可以获得与说意思最接近的前n个词。在本project中，我们设置n=50.
    我们利用Stanfordcorenlp 作为工具进行分词，能够获得word_list(句子进行分词后的词列表)，parser_list(分词结果及其对应句子成分)，
    Dep_Parse(分词后，不同单词之间在原句子中的逻辑关系))
'''

def find_spoken_word_id_and_sub(spoken_words,word_list,parser_list):
    """
        在此函数中，我们输入spoken_words, word_list, parser_list，并输出句子主谓关系中，谓语属于spoken_words 的 主语和谓语在word_list中的位置。
    """
    #取得 新闻文章中 包含SBV关系 并且V表示的是说的意思 的主语和谓语的位置
    id_list = []
    for ele in parser_list:
        relation = ele[0]
        idx = (ele[2]-1, ele[1]-1)
        if  (relation == 'nsubj') and (word_list[idx[1]] in spoken_words):
            #'nsubj' 表示对应的两个词语在句子中是主谓关系. 
            id_list.append(idx)
    return id_list

'''# get_next_sentence '''
def get_next_sentence(index,news):
    '''
        函数 get_next_sentence 能够获取index之后的下一个句子，并判断何时句子结束以及是否有下一句。
    '''
    if index >= (len(news)-1):
        return False
    else:
        begin2 = float("inf")
        end2    = float("inf")
        begin2 = index + 1
        stop1 = news[index+1:].find('。')
        stop2 = news[index+1:].find('！')
        stop3 = news[index+1:].find('？')
        stop4 = news[index+1:].find('......')
        stop_list = [stop for stop in [stop1,stop2,stop3,stop4] if stop != -1]
        if not stop_list:
            return False
        else:
            end2 = min(stop_list)
            result2 = news[(begin2+2):index+end2+1]
    return result2,end2+begin2

def get_sentence_distance(sentence1,sentence2):
    '''
        get_sentence_distance 基于不同的度量标准，获得句子之间的距离。
        此处我们使用余弦距离，获得不同句子之间的距离。
    '''
    #计算句子间的距离 这里使用余弦距离 句子的embedding使用的是句子中词向量相加
    word_list1 = nlp.word_tokenize(sentence1)
    word_list2 = nlp.word_tokenize(sentence2)
    vec_1 = 0
    vec_2 = 0
    # 获得句子一的表示。
    for i in range(len(word_list1)):
        if word_list1[i] in new_model.wv.vocab:
            vec_1 += new_model.wv[word_list1[i]]

    # 获得句子二的表示。
    for j in range(len(word_list2)):
        if word_list2[j] in new_model.wv.vocab:
            vec_2 += new_model.wv[word_list2[j]]

    return np.dot(vec_1,vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2))

nlp = StanfordCoreNLP(currentDirectory + '/lang_model/stanford-corenlp-full-2018-10-05',lang='zh')
new_model =Word2Vec.load(currentDirectory + '/lang_model/lang_model')
spoken_words = [word for word, i in new_model.wv.most_similar('说', topn = 50)]
spoken_words.append('说')

def get_opinions(news,debug = False):
    '''
        get_opinions 函数通过对人物不同表达观点方式的学习，获得本信息中心人物表达的观点。
        输入为原始文本，输出为输入文本中不同人物及其对应的观点。
    '''
    word_list = nlp.word_tokenize(news)
    parser_list = nlp.pos_tag(news)
    Dep_Parse = nlp.dependency_parse(news)

    idx = find_spoken_word_id_and_sub(spoken_words,word_list,Dep_Parse)
    idx_sub = [ele[0] for ele in idx]
    idx_verb = [ele[1] for ele in idx]

    print(idx_sub)

    if debug:
        print(word_list)
        print(Dep_Parse)
        print(idx)

    result = ''
    table = defaultdict(list)
    for i in range(len(idx_verb)):
        result += '\n' +'{}: '.format(parser_list[idx_sub[i]][0])
        key = parser_list[idx_sub[i]][0]
        begin = float("inf")
        end = float("inf")
        index = len("".join(word_list[:idx_sub[i]]))

        ###type 1: 寻找 说的内容在说前的句子 诸如: "......。"XX说。
        if parser_list[idx_verb[i]+1][0]  == '。':
            #index = len("".join(word_list[:idx_sub[i]]))
            # if parser_list[idx_sub[i]-1][0] == '”':
            print('block 1')
            
            begin = news[:index][::-1].find('“')
            end = news[:index][::-1].find('”')
            if begin == -1 or end == -1:
                continue
            table[key].append(news[:index][::-1][end+1:begin][::-1])
            #result = result + news[:index][::-1][end+1:begin][::-1]
            
        ###type 2: 寻找说的前后都有 说的内容的句子， 诸如：”.....。“ XX说，”......。"
        elif parser_list[idx_verb[i]+1][0] == '，' and parser_list[idx_verb[i]+2][0] == '“':
            print('block 2')
            index1 = len("".join(word_list[:idx_sub[i]]))
            begin1 = float("inf")
            end1 = float("inf")
            begin1 = news[:index1][::-1].find("“")
            end1 = news[:index1][::-1].find("”")
            result1 = news[:index1][::-1][end1:begin1+1][::-1]

            begin2 = float("inf")
            end2 = float("inf")
            begin2 = news[index1:].find("“")
            end2 = news[index1:].find('”')
            result2 = news[index1:][begin2:end2+1]

            table[key].append(result1+result2)
            #result += result1 + result2

        ###type 3: 寻找只有说的后面有句子，且形式为，XX说：“....。”
        elif parser_list[idx_verb[i]+1][0] == '：' and parser_list[idx_verb[i]+2][0] == '“':
            print('block 3')
            begin = news[index:].find("“")
            end   = news[index:].find('”')

            table[key].append(news[index:][begin:end+1])
            result += news[index:][begin:end+1]
        ###type 4: 寻找只有说的后面有句子，但形式为，XX说, .....。以及句号后可能还跟有句子。
        elif parser_list[idx_verb[i]+1][0] == '，':
            print('block 4')
            sim = 1
            result0, index2 = get_next_sentence(index+2, news)
           
            table[key].append(result0)
            #result += result0

            while True:
                if get_next_sentence(index2, news) != False:
                    sim = get_sentence_distance(result0,get_next_sentence(index2, news)[0])
                    print(sim)
                    if debug:
                        print(sim)
                    if sim>0.5:
                        result0, index2 = get_next_sentence(index2, news)
                        #result += '\n' + result0
                        table[key].append(result0)
                    else:
                        break
                else:
                    break



    return table

def print_table(table):
    for key, val in table.items():
        print("========")
        print("key:{}, val:{}".format(key, val))
        print("========\n")

def test():
    '''
        test 函数执行测试过程，选择不同文本，并输出人物观点。
    '''
    txt1 = '“这件事儿就这么算了吧。”他说。'
    # result should be 
    txt2 = '“这件事儿就这么算了吧。”他说，“先生说了，还是好好想想接下来要怎么做。”她回答说：”那行吧，先听你的吧。“'
    txt3 = '李小云高兴的说：”学堂的先生说，灾变之前有眼镜这种东西，现在其实也有，只不过在避难壁垒里面，有了这个东西就算近视了也不怕。“'
    txt4 = '他说，明天必须来上课。因为明天教授要点名。从古到今，打着正义旗号二进行的战争不胜枚举。太行山上王屋情。'
    test_txt = '李小云认为：”学堂的先生说，灾变之前有眼镜这种东西，他还说现在其实也有，只不过在避难壁垒里面，有了这个东西就算近视了也不怕。“张天一说：”明天必须来上课。因为明天教授要点名。“从古到今，打着正义旗号二进行的战争不胜枚举。太行山上王屋情。'
    for msg in [txt1, txt2, txt3, txt4, test_txt]:
        table = get_opinions(msg)
        print_table(table)
        

