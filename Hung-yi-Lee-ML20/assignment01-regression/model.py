import numpy as np
import pandas as pd
import csv

def loadCSV(csvFile, encoding="big5", header=0, starCol=3):
    
    csvData = pd.read_csv(csvFile, header=header, encoding=encoding)
    csvData = csvData.iloc[:, starCol:]    # start from 4th column
    csvData[csvData == "NR"] = 0     # correct rainfall 
    data = csvData.to_numpy()

    return data


def preProcess(data):

    yearData = {}

    for month in range(12):
        monthData = np.empty([18, 480])
        for day in range(20):
            monthData[:, day*24:(day+1)*24] = data[18*(month*20+day):18*(month*20+day+1),:]
        
        yearData[month] = monthData    # shape: 18*480

    return yearData


def extractFeature(yearData):

    x = np.empty([12*471, 18*9], dtype=float)
    y = np.empty([12*471, 1], dtype=float)

    for month in yearData:
        for row in range(471):
            x[month*471+row] = yearData[month][:, row:row+9].T.reshape((18*9)) # 18 features/day in sequence
            y[month*471+row] = yearData[month][9,row+9]
    
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)

    x = (x - mean_x)/std_x

    return x, y, mean_x, std_x


def splitData(x, y):
    # TODO
    return x, y


def train(x, y, l=100, iter=100000, eps=0.000000001):

    dim = x.shape[1]
    amount = x.shape[0]
    adagrad = np.zeros([dim+1, 1])
    w = np.zeros([dim+1,1])
    x = np.concatenate((x, np.ones([amount, 1])), axis=1)

    
    for iter in range(iter):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)

        if iter % 1000 == 0:
            print("iteration: {0}, average loss: {1}".format(iter, loss))
        
        grad = 2 * np.dot(x.T, np.dot(x, w) - y)
        adagrad += np.square(grad)


        w = w - l * grad / np.sqrt(adagrad + eps)


    # SGD
    # l = 0.0005
    # iter = 1000
    # for iter in range(iter):
    #     loss = np.sqrt(np.average(np.square(np.dot(x, w) - y)))
    #     if iter % 100 == 0:
    #         print("iteration: {0}, average loss: {1}".format(iter, loss))

    #     for i in range(len(x)):
    #         grad = (np.dot(x[i], w) - y[i]) * x[i].reshape([x[i].shape[0], 1])
    #         w = w - l * grad

    return x, w


def test(csvFile, w, mean_x, std_x):
    testRaw = loadCSV(csvFile, header=None, starCol=2)
    testX = np.empty([240, 18*9], dtype=float)

    for i in range(240):
        
        testX[i] = testRaw[i*18 : (i+1)*18, :].T.reshape((18*9))
    

    testX = (testX - mean_x)/std_x

    testX = np.concatenate((testX, np.ones([testX.shape[0],1])), axis=1)

    return np.dot(testX, w)


def outputCSV(labelData):
    with open("submit.csv", mode="w", newline="") as submitFile:
        csv_writer = csv.writer(submitFile)
        header = ['id', 'value'] 
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), labelData[i][0]]
            csv_writer.writerow(row)
            


if __name__ == "__main__":
    data = loadCSV("./data/train.csv")
    yearData = preProcess(data)
    x, y, mean_x, std_x = extractFeature(yearData)
    x, w = train(x, y)
    predict = test("./data/test.csv", w, mean_x, std_x)
    outputCSV(predict)
    



