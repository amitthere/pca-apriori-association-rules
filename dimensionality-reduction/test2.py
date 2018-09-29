import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import sys


# Function to calculate PCA
def CalculatePCA(pdata):
    raj = pdata
    #print(raj)
    #sandesh = pdata.T
    #print(sandesh.shape)
    cv_mat = np.cov(pdata.T)
    #print(cv_mat)
    eig_val, eig_vec = np.linalg.eigh(cv_mat)
    #print(eig_val)
    #print('_________')
    #print(eig_vec)
    #print('_________')
    eig_vec = eig_vec.transpose()
    d = dict()
    for i in range(eig_vec.shape[1]):
        d[eig_val[i]] = eig_vec[i]
    eig_mat = sorted(d.items(), key=operator.itemgetter(0), reverse=True)
    eig_mat = eig_mat[:2]
    #print(eig_mat)
    #print("k", eig_mat)
    dataPCA = np.concatenate((eig_mat[0][1][:, None], eig_mat[1][1][:, None]), axis=1)
    Y = pdata.dot(dataPCA)
    print(Y)
    return Y


# Function to calculate SVD
def CalculateSVD(sdata):
    u, s, v = np.linalg.svd(sdata.T)
    u = u.transpose()
    u = u[:2]
    dataSVD = np.concatenate((u[0][:, None], u[1][:, None]), axis=1)
    W_SVD = sdata.dot(dataSVD)
    return W_SVD


# Function to calculate TNSE
def CalculateTNSE(tdata):
    u_tnse = TSNE(n_components=2).fit_transform(tdata.T)
    u_tnse = u_tnse.transpose()
    u_tnse = u_tnse[:2]
    dataTNSE = np.concatenate((u_tnse[0][:, None], u_tnse[1][:, None]), axis=1)
    W_TNSE = tdata.dot(dataTNSE)
    return W_TNSE


def main():
    # Getting command line input data from user
    # fname = sys.argv[1]
    pddata = pd.read_csv('../data/pca_a.txt', sep='\t', header=None)
    # fname = fname.split("/")[-1]
    ncols = len(pddata.columns)
    #print(ncols)
    data = pddata.iloc[:, :-1]
    #print(data.shape)
    data = data.values
    #print(data.shape)
    origdata = data.copy()
    data -= data.mean(axis=0)
    #print(data)
    # Running for file pca_a.txt - The PCA Matrix is stored in the variable data.
    Y = CalculatePCA(data)


# Plotting Scatter Plot for the returned data
# xval = pd.DataFrame(Y)[0]
# yval = pd.DataFrame(Y)[1]
# lbls = set(pddata[ncols-1])
# fig1 = plt.figure(1)
# for lbl in lbls:
#    cond = pddata[ncols-1] == lbl
#    plt.plot(xval[cond], yval[cond], linestyle='none', marker='o', label=lbl)

# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(numpoints=1)
# plt.subplots_adjust(bottom=.20, left=.20)
# fig1.suptitle("Algorithm - PCA, Text File - "+fname[:-4],fontsize=20)
# fig1.savefig("PCA_"+fname+".png")
# plt.show()

# Calling SVD
# SVDData = CalculateSVD(origdata)

# Plotting SVD
# X_SVD = pd.DataFrame(SVDData)[0]
# Y_SVD = pd.DataFrame(SVDData)[1]
# lbls = set(pddata[ncols-1])
# fig2 = plt.figure(2)
# for lbl in lbls:
#    cond = pddata[ncols-1] == lbl
#    plt.plot(X_SVD[cond], Y_SVD[cond], linestyle='none', marker='o', label=lbl)

# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(numpoints=1)
# plt.subplots_adjust(bottom=.20, left=.20)
# fig2.suptitle("Algorithm - SVD, Text File - "+fname[:-4],fontsize=20)
# fig2.savefig("SVD_"+fname+".png")
# plt.show()

# Calling TNSE
# TNSEData = CalculateTNSE(origdata)

# Plotting TNSE
# X_TNSE = pd.DataFrame(TNSEData)[0]
# Y_TNSE = pd.DataFrame(TNSEData)[1]
# lbls = set(pddata[ncols-1])
# fig3 = plt.figure(3)
# for lbl in lbls:
#    cond = pddata[ncols-1] == lbl
#    plt.plot(X_TNSE[cond], Y_TNSE[cond], linestyle='none', marker='o', label=lbl)

# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(numpoints=1)
# plt.subplots_adjust(bottom=.20, left=.20)
# plt.xticks(rotation=45)
# fig3.suptitle("Algorithm - TNSE, Text File - "+fname[:-4],fontsize=20)
# fig3.savefig("TNSE_"+fname+".png")
# plt.show()


if __name__ == '__main__':
    main()
