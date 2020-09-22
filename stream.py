from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.neighbors import KNeighborsClassifier


df=pd.read_csv(r'https://raw.githubusercontent.com/Giovann18/MLSTREAMLIT/master/train.csv')
df=df.rename(columns={"SibSp":"Number of Sibilings","Parch":"Number of Children"})
st.title('Hello')
st.header('Classification Problem - Titanic Dataset')
lista_target=st.sidebar.multiselect('Scegli le variabili predittore',['Pclass','Age','Number of Sibilings',
                                                              'Number of Children','Fare'],default=['Age','Fare'])
y=df['Survived']
X=df[lista_target]
#X=df[['Age','Fare','Pclass']]
X=X.fillna(X.mean())



if st.checkbox('Show full data'):
    st.write(df)
sample=st.sidebar.slider('Number of Samples in Training Set',0,890,600)
X_train=X[:sample]
X_test=X[sample:]
y_train=y[:sample]
y_test=y[sample:]
df_train=pd.concat([y_train,X_train],axis=1)
df_test=pd.concat([y_test,X_test],axis=1)

if st.checkbox('Show train and test data'):
    st.write('Training Set')
    st.write(df_train)
    st.write('Test Set')
    st.write(df_test)

assex=st.sidebar.selectbox("Scegli la variabile da mostrare sull'asse x",
                       (X.columns))
assey=st.sidebar.selectbox("Scegli la variabile da mostrare sull'asse y",
                       (X.columns))


if st.checkbox('Show Training Set Plot'):
    st.subheader('Actual Training Set Plot')
    #plt.scatter(df_train['Age'],df_train['Fare'],c=df_train['Survived'])
    plt.scatter(df_train[assex],df_train[assey],c=df_train['Survived'])
    plt.xlabel(assex)
    plt.ylabel(assey)
    st.pyplot()
    plt.clf()

st.subheader('Actual Test Set Plot')
plt.scatter(df_test[assex],df_test[assey],c=df_test['Survived'])
plt.xlabel(assex)
plt.ylabel(assey)
st.pyplot()
plt.clf()
algoritmo=st.selectbox("Scegli l'algoritmo di classificazione",
                       ('Naive Bayes','Random Forest','KNN'))



if algoritmo =='Naive Bayes':
    gnb=GaussianNB()
    gnb.fit(X_train,y_train)
    if gnb.score(X_test,y_test)<0.6:
        st.error("Accuracy:{0:.2f}".format(gnb.score(X_test,y_test)))
    else:
        st.success("Accuracy:{0:.2f}".format(gnb.score(X_test,y_test)))
    #st.write('*Accuracy:* ',"{0:.2f}".format(gnb.score(X_test,y_test)))
    st.subheader('Predicted Naive Bayes Test Set Plot')
    plt.scatter(X_test[assex],X_test[assey],c=gnb.predict(X_test))
    plt.xlabel(assex)
    plt.ylabel(assey)
    st.pyplot()
    plt.clf()
    st.subheader('Simulazioni')
    predictorss=[]
    for c in X_test.columns:
        if c=='Age' or c=='Fare':
            c=st.slider('Scegli un valore di '+c,X_test[c].min(),X_test[c].max(),X_test[c].mean())
            predictorss.append(c)
        else:    
            c=st.selectbox(c,
                       (X_test[c].unique()))
            predictorss.append(c)
    preddd=gnb.predict([predictorss])
    if preddd == 0:
        st.markdown('La persona è morta :(')
    else:
        st.markdown('La persona è viva :)')
        
    
if algoritmo =='Random Forest':
    split_type=st.radio('Split Criterion',('gini','entropy'))
    rfc=RandomForestClassifier(criterion=split_type)
    rfc.fit(X_train,y_train)
    if rfc.score(X_test,y_test)<0.6:
        st.error("Accuracy:{0:.2f}".format(rfc.score(X_test,y_test)))
    else:
        st.success("Accuracy:{0:.2f}".format(rfc.score(X_test,y_test)))
    #st.write('*Accuracy:* ',"{0:.2f}".format(rfc.score(X_test,y_test)))
    st.subheader('Predicted Random Forest Test Set Plot')
    plt.scatter(X_test[assex],X_test[assey],c=rfc.predict(X_test))
    plt.xlabel(assex)
    plt.ylabel(assey)
    st.pyplot()
    plt.clf()
    st.subheader('Simulazioni')
    predictorss=[]
    for c in X_test.columns:
        if c=='Age' or c=='Fare':
            c=st.slider('Scegli un valore di '+c,X_test[c].min(),X_test[c].max(),X_test[c].mean())
            predictorss.append(c)
        else:    
            c=st.selectbox(c,
                       (X_test[c].unique()))
            predictorss.append(c)
    preddd=rfc.predict([predictorss])
    if preddd == 0:
        st.markdown('La persona è morta :(')
    else:
        st.markdown('La persona è viva :)')

if algoritmo =='KNN':
    nn=st.slider('Number of Neighbors', 1, 10, 3)
    neigh = KNeighborsClassifier(n_neighbors=nn)
    neigh.fit(X_train,y_train)
    if neigh.score(X_test,y_test)<0.6:
        st.error("Accuracy:{0:.2f}".format(neigh.score(X_test,y_test)))
    else:
        st.success("Accuracy:{0:.2f}".format(neigh.score(X_test,y_test)))
    #st.write('*Accuracy:* ',"{0:.2f}".format(neigh.score(X_test,y_test)))
    st.subheader('Predicted KNN Test Set Plot')
    plt.scatter(X_test[assex],X_test[assey],c=neigh.predict(X_test))
    plt.xlabel(assex)
    plt.ylabel(assey)
    st.pyplot()
    plt.clf()
    st.subheader('Simulazioni')
    predictorss=[]
    for c in X_test.columns:
        if c=='Age' or c=='Fare':
            c=st.slider('Scegli un valore di '+c,X_test[c].min(),X_test[c].max(),X_test[c].mean())
            predictorss.append(c)
        else:    
            c=st.selectbox(c,
                       (X_test[c].unique()))
            predictorss.append(c)
    preddd=neigh.predict([predictorss])
    if preddd == 0:
        st.markdown('La persona è morta :(')
    else:
        st.markdown('La persona è viva :)')
        
