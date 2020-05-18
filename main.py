import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class bayesian:
    def __init__(self, trainData, testData):
        self.trainData = trainData
        self.testData = testData
        # priori probability
        self.p1 = len(
            self.testData[self.testData['Label'] == 1])/len(self.testData)
        self.p2 = len(
            self.testData[self.testData['Label'] == 2])/len(self.testData)
        self.p3 = len(
            self.testData[self.testData['Label'] == 3])/len(self.testData)
        # mean
        self.M1 = self.trainData[self.trainData['Label'] == 1].drop(
            ['Label'], axis=1).mean().to_numpy()
        self.M2 = self.trainData[self.trainData['Label'] == 2].drop(
            ['Label'], axis=1).mean().to_numpy()
        self.M3 = self.trainData[self.trainData['Label'] == 3].drop(
            ['Label'], axis=1).mean().to_numpy()
        # covariance matric Σ
        self.sigma1 = self.trainData[self.trainData['Label'] == 1].drop(
            ['Label'], axis=1).cov().to_numpy()
        self.sigma2 = self.trainData[self.trainData['Label'] == 2].drop(
            ['Label'], axis=1).cov().to_numpy()
        self.sigma3 = self.trainData[self.trainData['Label'] == 3].drop(
            ['Label'], axis=1).cov().to_numpy()

    def mu(self, M1, M2, sigma1, sigma2):
        u12 = 1/8*np.dot(np.dot(np.transpose((M2 - M1)), np.linalg.inv((sigma1+sigma2)/2)), (M2 - M1)) + \
            1/2*np.log(np.linalg.det((sigma1+sigma2)/2) /
                       np.sqrt(np.linalg.det(sigma1)*np.linalg.det(sigma2)))
        return u12

    # https://en.wikipedia.org/wiki/Bhattacharyya_distance
    def upperBound(self, p1, p2, M1, M2, sigma1, sigma2):
        # caculate error
        e12 = np.sqrt(p1*p2)*np.exp(-self.mu(M1, M2, sigma1, sigma2))
        return e12

    def show_upperBound(self):
        print('Label 1,2 upper bound: ', self.upperBound(
            self.p1, self.p2, self.M1, self.M2, self.sigma1, self.sigma2))
        print('Label 1,3 upper bound: ', self.upperBound(
            self.p1, self.p3, self.M1, self.M3, self.sigma1, self.sigma3))
        print('Label 2,3 upper bound: ', self.upperBound(
            self.p2, self.p3, self.M2, self.M3, self.sigma2, self.sigma3))

    # https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation
    def classifier(self, X):
        # caculate portability
        probability1 = -np.log(self.p1) + 1/2*np.dot(np.dot(np.transpose(X-self.M1),
                                                            np.linalg.inv(self.sigma1)), (X-self.M1)) + 1/2*np.log(np.linalg.det(self.sigma1))
        probability2 = -np.log(self.p2) + 1/2*np.dot(np.dot(np.transpose(X-self.M2),
                                                            np.linalg.inv(self.sigma2)), (X-self.M2)) + 1/2*np.log(np.linalg.det(self.sigma2))
        probability3 = -np.log(self.p3) + 1/2*np.dot(np.dot(np.transpose(X-self.M3),
                                                            np.linalg.inv(self.sigma3)), (X-self.M3)) + 1/2*np.log(np.linalg.det(self.sigma3))

        minimum = min(probability1, probability2, probability3)
        if minimum == probability1:
            return 1
        if minimum == probability2:
            return 2
        if minimum == probability3:
            return 3

    def accuracy(self):
        x_answer = []
        y_answer = []
        px_answer = []
        py_answer = []
        for i in self.trainData.to_numpy():
            temp = i[1:]
            x_answer.append(i[0])
            px_answer.append(self.classifier(temp))

        for i in self.testData.to_numpy():
            temp = i[1:]
            y_answer.append(i[0])
            py_answer.append(self.classifier(temp))

        score_x = 0
        score_y = 0
        for index in range(len(px_answer)):
            if x_answer[index] == px_answer[index]:
                score_x += 1
            if y_answer[index] == py_answer[index]:
                score_y += 1

        print("Pridict train accuracy: ", score_x/len(px_answer))
        print("Pridict test accuracy: ", score_y/len(py_answer))


if __name__ == '__main__':
    df = pd.read_csv('data/wine.data', header=None)
    df.columns = ['Label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                  'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    # split data
    df_train, df_test = train_test_split(df, test_size=0.5)

    wine = bayesian(df_train, df_test)
    wine.accuracy()
    wine.show_upperBound()
