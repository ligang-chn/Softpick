import math
import datetime
import sys
import numpy as np


class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name#训练文件
        self.predict_file = test_file_name#预测文件
        self.predict_result_file = predict_result_file_name#预测结果
        self.max_iters = 1000#最大迭代次数
        self.rate = 0.01#学习率
        self.feats = []
        self.labels = []
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.weight = []
        
    def train_test_split(X, y, test_ratio=0.2, seed=None):
        """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
        assert X.shape[0] == y.shape[0], \
            "the size of X must be equal to the size of y"
        assert 0.0 <= test_ratio <= 1.0, \
            "test_ration must be valid"
        if seed:
            np.random.seed(seed)
        shuffled_indexes = np.random.permutation(len(X))

        test_size = int(len(X) * test_ratio)
        test_indexes = shuffled_indexes[:test_size]
        train_indexes = shuffled_indexes[test_size:]

        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        return X_train, X_test, y_train, y_test


    #加载数据
    def loadDataSet(self, file_name, label_existed_flag):
        feats = []
        labels = []
        fr = open(file_name)
        lines = fr.readlines()
        for line in lines:
            temp = []
            allInfo = line.strip().split(',')
            dims = len(allInfo)
            if label_existed_flag == 1:
                for index in range(dims-1):
                    temp.append(float(allInfo[index]))
                feats.append(temp)
                labels.append(float(allInfo[dims-1]))
            else:
                for index in range(dims):
                    temp.append(float(allInfo[index]))
                feats.append(temp)
        fr.close()
        feats = np.array(feats)
        labels = np.array(labels)
        X_train,X_test,y_train,y_test=train_test_split(feats,labels,0.3)
        return X_train,X_test,y_train,y_test


    #加载训练数据
    def loadTrainData(self):
        self.feats, self.labels = self.loadDataSet(self.train_file, 1)
    #加载测试数据
    def loadTestData(self):
        self.feats_test, self.labels_predict = self.loadDataSet(
            self.predict_file, 0)

    #保存预测结果
    def savePredictResult(self):
        print(self.labels_predict)
        f = open(self.predict_result_file, 'w')
        for i in range(len(self.labels_predict)):
            f.write(str(self.labels_predict[i])+"\n")
        f.close()



    #激励函数
    def sigmod(self, x):
        return 1/(1+np.exp(-x))

    def printInfo(self):
        print(self.train_file)
        print(self.predict_file)
        print(self.predict_result_file)
        print(self.feats)
        print(self.labels)
        print(self.feats_test)
        print(self.labels_predict)

    def initParams(self):
        self.weight = np.ones((self.param_num,), dtype=np.float)

    def compute(self, recNum, param_num, feats, w):
        return self.sigmod(np.dot(feats, w))

    #误差率
    def error_rate(self, recNum, label, preval):
        return np.power(label - preval, 2).sum()

    def predict(self):
        self.loadTestData()
        preval = self.compute(len(self.feats_test),
                              self.param_num, self.feats_test, self.weight)
        self.labels_predict = (preval+0.5).astype(np.int)
        self.savePredictResult()

    def train(self):
        self.loadTrainData()
        recNum = len(self.feats)
        self.param_num = len(self.feats[0])
        self.initParams()
        ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S,f'
        for i in range(self.max_iters):
            preval = self.compute(recNum, self.param_num,
                                  self.feats, self.weight)
            sum_err = self.error_rate(recNum, self.labels, preval)
            if i%30 == 0:
                print("Iters:" + str(i) + " error:" + str(sum_err))
                theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
                print(theTime)
            err = self.labels - preval
            delt_w = np.dot(self.feats.T, err)
            delt_w /= recNum
            self.weight += self.rate*delt_w


def print_help_and_exit():
    print("usage:python3 main.py train_data.txt test_data.txt predict.txt [debug]")
    sys.exit(-1)


def parse_args():
    debug = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'debug':
            print("test mode")
            debug = True
        else:
            print_help_and_exit()
    return debug


if __name__ == "__main__":
    debug = parse_args()
    train_file =  "./data/train_data.txt"
    test_file = "./data/test_data.txt"
    predict_file = "./projects/student/result.txt"
    lr = LR(train_file, test_file, predict_file)
    lr.train()
    lr.predict()

    if debug:
        answer_file ="./projects/student/answer.txt"
        f_a = open(answer_file, 'r')
        f_p = open(predict_file, 'r')
        a = []
        p = []
        lines = f_a.readlines()
        for line in lines:
            a.append(int(float(line.strip())))
        f_a.close()

        lines = f_p.readlines()
        for line in lines:
            p.append(int(float(line.strip())))
        f_p.close()

        print("answer lines:%d" % (len(a)))
        print("predict lines:%d" % (len(p)))

        errline = 0
        for i in range(len(a)):
            if a[i] != p[i]:
                errline += 1

        accuracy = (len(a)-errline)/len(a)
        print("accuracy:%f" %(accuracy))