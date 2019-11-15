import data
import algorithm as al
import numpy as np
from sklearn.model_selection import KFold

x = data.x
x = (x - np.mean(x, axis = 0))/np.std(x, axis = 0)
t = data.t

scores_mse = []
scores_r_2 = []
cv = KFold(n_splits=10, random_state=11152019, shuffle=True)
for train_index, test_index in cv.split(x):
    x_train, x_test, t_train, t_test = x[train_index], x[test_index], t[train_index], t[test_index]

    #copy main.py
    model = al.algorithm(eta = 0.0002, c = 0)
    model.fit(x_train, t_train)
    print(model.w)
    print(model.b)

    y_predict = model.predict(x_test)
    print(y_predict)

    mse = np.dot( (t_test - y_predict),  (t_test - y_predict) )/len(t_test)
    print("Mean square error = %.2f" %(mse))

    tot = np.dot( t_test - np.mean(t_test), t_test - np.mean(t_test) )/len(t_test)
    r_2 = 1 - ( mse / tot)
    print("R square score = %.2f" %(r_2))

    scores_mse.append(mse)
    scores_r_2.append(r_2)

print('average mse = %.2f' %(np.mean(scores_mse)))
print('average r_2 = %.2f' %(np.mean(scores_r_2)))


'''
x = data.x
x = (x - np.mean(x, axis = 0))/np.std(x, axis = 0)
t = data.t

scores_mse = []
scores_r_2 = []
cv = KFold(n_splits=10, random_state=11152019, shuffle=True)
for train_index, test_index in cv.split(x):
    x_train, x_test, t_train, t_test = x[train_index], x[test_index], t[train_index], t[test_index]

    model = al.algorithm()
    model.fit(x_train, t_train)
    print(model.w)
    print(model.b)

    y_predict = model.predict(x_test)

    mse = np.dot( (t_test - y_predict)**2,  (t_test - y_predict)**2 )/len(t_test)
    print("Mean square error = %.2f" %(mse))

    tot = np.dot( t_test - np.mean(t_test), t_test - np.mean(t_test) )/len(t_test)
    r_2 = 1 - ( mse / tot)
    print("R square score = %.2f" %(r_2))

    scores_mse.append(mse)
    scores_r_2.append(r_2)
    print("#"*50)

print('average mse = %.2f' %(np.mean(scores_mse)))
print('average r_2 = %.2f' %(np.mean(scores_r_2)))

'''