

class A(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, output_dir = None):
        self.output_dir = output_dir



class B(A):
    def look(self):
        print('AAA')


    def look2(self):
        print(self.output_dir)



def C():
    print('c')



def main():
    bb = B()
    bb.look()
    bb2 = B(output_dir='123')
    bb2.look2()


def check_data():
    import csv
    import nltk
    import numpy as np
    from nltk.tokenize import sent_tokenize
    import json
    np.random.seed(1)
    #path = './datasets/amazon-2/amazon_review_polarity_csv/train.csv'
    path = './datasets/dbpedia/dbpedia_csv/train.csv'
    json_result_path = './datasets/ins/dbpedia/train.json'
    with open(path, 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr, delimiter=',', quotechar='"')
        selected_paras = []
        for i, line in enumerate(reader):
            #if i==200:
            #    break
            title = line[0]
            para = line[-1]
            id = "ins_dbpedia_%d"%i
            sens = sent_tokenize(para)
            if len(sens)>4:
                selected_paras.append((para,title, id))
    print(len(selected_paras))
    exit()


    ins_sen = []

    for para, title, id in selected_paras:
        sens = sent_tokenize(para)
        # TODO: remove the -1 after len(sens)
        miss_idx = np.random.randint(0, len(sens) - 1)
        miss_sen = sens[miss_idx]
        #rem_para = '<INS> ' + ' <INS> '.join(sens[:miss_idx] + sens[miss_idx+1:]) + ' <INS>'
        prev_str = ' '.join(sens[:miss_idx])
        if miss_idx == 0:
            answer_start = 0
        else:
            answer_start = len(prev_str) + len(' ')
        answer_text = sens[miss_idx+1].split(' ')[0]
        rem_para = ' '.join(sens[:miss_idx] + sens[miss_idx+1:])
        assert(rem_para[answer_start] == answer_text[0])
        ins_sen.append((title, rem_para, miss_sen, miss_idx, id, answer_start, answer_text))

    #for title, para, sen, miss_idx, id, answer_start, answer_text in ins_sen:
    #    print("PARA: \n{}\nSEN: \n{}\n\n".format(para, sen))

    json_dict = {"data":[]}
    for title, para, sen, miss_idx, id, answer_start, answer_text in ins_sen:
        context = para
        question = sen
        #answer_text = ""
        #answer_start = 0

        article_dict = dict()
        article_dict['title'] = title
        qas_dict = [{"id":id, "question": question, "answers":[{"text":answer_text, "answer_start":answer_start}]}]
        article_dict["paragraphs"] = [{"context": context, "qas":qas_dict}]


        json_dict["data"].append(article_dict)
    with open(json_result_path, 'w') as fw:
        json.dump(json_dict, fw)







if __name__ == '__main__':
    check_data()

