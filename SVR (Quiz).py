#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# 1.โหลด csv เข้าไปใน Python Pandas

# In[2]:


df = pd.read_csv('../Desktop/DataCamp/HousePricePrediction.csv')
df


# 2. เขียนโค้ดแสดง หัว10แถว ท้าย10แถว และสุ่ม10แถว

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.sample(10)


# 3. เช็คว่ามีข้อมูลที่หายไปไหม สามารถจัดการได้ตามความเหมาะสม

# In[6]:


df.isnull().any()


# 4. ใช้ info และ describe อธิบายข้อมูลเบื้องต้น

# In[7]:


df.info()


# In[8]:


df.describe()


# 5. ใช้ pairplot ดูความสัมพันธ์เบื้องต้น

# In[9]:


sns.pairplot(df)


# 6. ใช้ displot เพื่อดูการกระจายของแต่ละคอลัมน์

# In[10]:


sns.distplot(df['price'])


# In[11]:


sns.distplot(df['bedrooms'])


# In[12]:


sns.distplot(df['bathrooms'])


# In[13]:


sns.distplot(df['sqft_living'])


# In[14]:


sns.distplot(df['sqft_lot'])


# In[15]:


sns.distplot(df['floors'])


# In[16]:


sns.distplot(df['condition'])


# In[17]:


sns.distplot(df['sqft_above'])


# In[18]:


sns.distplot(df['sqft_basement'])


# In[19]:


sns.distplot(df['yr_built'])


# In[20]:


sns.distplot(df['yr_renovated'])


# 7. ใช้ heatmap ดูความสัมพันธ์ของคอลัมน์ที่สนใจ

# In[21]:


sns.heatmap(df.corr())


# 8. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation สูงสุด

# In[22]:


sns.scatterplot(data = df, y = 'sqft_living', x ='sqft_above')


# 9. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation ต่ำสุด

# In[23]:


sns.scatterplot(data = df, y = 'bedrooms', x ='condition')


# 10. สร้าง histogram ของ price

# In[24]:


plt.hist(df['price'])


# 11. สร้าง box plot ของราคา

# In[25]:


sns.boxplot(df['price'], orient = 'v')


# 12. สร้าง train/test split ของบ้าน สามารถลองทดสอบ 70:30, 80:20, 90:10 ratio ได้ตามใจชอบ

# In[116]:


df = df.sort_values(by = ['sqft_living'])
df.head()


# In[81]:


X = df['sqft_living']
y = df['price']


# In[82]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[83]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)


# 13. ทำ Data Transformation และ Data Scaling

# In[84]:


#data transformation 
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


# In[85]:


#data scaling
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()

X1_train = sc_X.fit_transform(X_train)
y1_train = sc_y.fit_transform(y_train)
X1_test = sc_X.fit_transform(X_test)
y1_test = sc_y.fit_transform(y_test)


# 14. เทรนโมเดลแบบ Linear และ rbf ของคู่ที่ผู้เรียนคิดว่าเหมาะสม
# (ให้เป็น 1 dependent VS 1 independent)

# 15. ทดสอบโมเดลวัดค่า MAE, MSE, RMSE

# In[86]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X1_train ,y1_train)


# In[89]:


predicted_linear = sc_y.inverse_transform(regressor.predict(X1_test))
predicted_linear


# In[90]:


print('MAE', metrics.mean_absolute_error(predicted_linear,sc_y.inverse_transform(y1_test)))
print('MSE', metrics.mean_squared_error(predicted_linear,sc_y.inverse_transform(y1_test)))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_linear,sc_y.inverse_transform(y1_test))))


# In[91]:


fig = plt.figure(figsize = (12,8))
plt.scatter(sc_X.inverse_transform(X1_test),sc_y.inverse_transform(y1_test), color = 'red')
plt.plot(X_test,predicted_linear,color = 'blue')
plt.title('SVR Linear')
plt.xlabel('sqft living')
plt.ylabel('Housing Price')
plt.show()


# In[92]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X1_train ,y1_train)


# In[93]:


predicted_rbf = sc_y.inverse_transform(regressor.predict(X1_test))
predicted_rbf


# In[94]:


print('MAE', metrics.mean_absolute_error(predicted_rbf,sc_y.inverse_transform(y1_test)))
print('MSE', metrics.mean_squared_error(predicted_rbf,sc_y.inverse_transform(y1_test)))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_rbf,sc_y.inverse_transform(y1_test))))


# 16. เทรนโมเดลแบบ Linear และ rbf ของทั้งหมด หรือ features ที่ผู้เรียนคิดว่าเหมาะสม
# (ให้เป็น 1 dependent variable VS many independent variables)
# Hint: คล้ายๆ Multiple Linear Regression แต่ต้องทำ Data Scaling กับทุก features

# In[95]:


X_multi = df[['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_basement']]
y_multi = df['price']


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size = 0.2, random_state = 100)


# In[97]:


y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)    #ไม่ reshape x 


# In[98]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()
X2_train = sc_X.fit_transform(X_train)
y2_train = sc_y.fit_transform(y_train)
X2_test = sc_X.fit_transform(X_test)
y2_test = sc_y.fit_transform(y_test)


# In[99]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X2_train,y2_train)


# 17. ทดสอบโมเดลวัดค่า MAE, MSE, RMSE

# In[100]:


predicted_linear2 = sc_y.inverse_transform(regressor.predict(X2_test))
predicted_linear2


# In[101]:


print('MAE', metrics.mean_absolute_error(predicted_linear2,sc_y.inverse_transform(y2_test)))
print('MSE', metrics.mean_squared_error(predicted_linear2,sc_y.inverse_transform(y2_test)))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_linear2,sc_y.inverse_transform(y2_test))))


# In[102]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X2_train,y2_train)


# In[103]:


predicted_rbf2 = sc_y.inverse_transform(regressor.predict(X2_test))
predicted_rbf2


# In[104]:


print('MAE', metrics.mean_absolute_error(predicted_rbf2,sc_y.inverse_transform(y2_test)))
print('MSE', metrics.mean_squared_error(predicted_rbf2,sc_y.inverse_transform(y2_test)))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_rbf2,sc_y.inverse_transform(y2_test))))


# 18. เทรนโมเดลแบบ Simple Linear Regression และ Multiple Linear Regression
# (สามารถเลือกคู่ได้ หรือ จะลองทุกคู่ก็ได้ ตามความคิดของผู้เรียน)
# 

# 19. ทดสอบโมเดลวัดค่า MAE, MSE, RMSE ของ Linear Regression
# 

# In[105]:


X = df['sqft_living']
y = df['price']


# In[106]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)


# In[108]:


X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


# In[109]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[110]:


print(lm.intercept_)
print(lm.coef_)


# In[111]:


predicted_SLR = lm.predict(X_test)
predicted_SLR 


# In[112]:


print('MAE', metrics.mean_absolute_error(predicted_SLR,y_test))
print('MSE', metrics.mean_squared_error(predicted_SLR,y_test))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_SLR,y_test)))


# In[113]:


fig = plt.figure(figsize = (12,8))
plt.scatter(X_test,y_test,color = 'blue', label = 'real price')
plt.plot(X_test,predicted_SLR,color = 'red', label = 'Linear regression price')
plt.xlabel('sqft living')
plt.ylabel('Housing price')
plt.title('the relationship between sqft living and housing price')
plt.legend()


# In[114]:


X_multi = df[['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_basement']]
y_multi = df['price']


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size = 0.2, random_state = 100)


# In[60]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[61]:


predicted_MLR = lm.predict(X_test)
predicted_MLR


# In[62]:


print('MAE', metrics.mean_absolute_error(predicted_MLR,y_test))
print('MSE', metrics.mean_squared_error(predicted_MLR,y_test))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_MLR,y_test)))


# 20. เปรียบเทียบผลลัพธ์ที่ดีที่สุดของ Linear Regression กับ SVR ว่าโมเดลไหนมีประสิทธิภาพมากกว่า

# In[63]:


#One independent - One dependent 
#Simple Linear Regression 
print('RMSE Simple Linear Regression', np.sqrt(metrics.mean_squared_error(predicted_SLR,y_test)))
#SVR linear 
print('RMSE SVR Linear', np.sqrt(metrics.mean_squared_error(predicted_linear,sc_y.inverse_transform(y1_test))))
#SVR rbf 
print('RMSE SVR rbf', np.sqrt(metrics.mean_squared_error(predicted_rbf,sc_y.inverse_transform(y1_test))))


# In[64]:


#Many independent - One dependent 
#Multiple Linear Regression 
print('RMSE Multiple Linear Regression', np.sqrt(metrics.mean_squared_error(predicted_MLR,y_test)))
#SVR linear 
print('RMSE SVR Linear', np.sqrt(metrics.mean_squared_error(predicted_linear2,sc_y.inverse_transform(y2_test))))
#SVR rbf 
print('RMSE SVR rbf', np.sqrt(metrics.mean_squared_error(predicted_rbf2,sc_y.inverse_transform(y2_test))))


# 21. สร้าง scatter plot และ prediction line ของ simple linear regression ที่ดีที่สุด และ SVR ( แบบ 1 dependent VS 1 independent) ที่ดีที่สุด

# กราฟอยู่ในข้อ 15,19
