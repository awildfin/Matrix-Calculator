from flask import Flask, render_template, request, jsonify
import numpy as np
import scipy.linalg

app = Flask(__name__)

def eigenvalues_and_vectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

def lu_decomposition(matrix):
    P, L, U = scipy.linalg.lu(matrix)
    return P, L, U

def cholesky_decomposition(matrix):
    try:
        L = np.linalg.cholesky(matrix)
        return L, L.T
    except np.linalg.LinAlgError:
        return None, None

def doolittle_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = matrix[i][k] - sum

        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (matrix[k][i] - sum) / U[i][i]
    return L, U

def crout_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j][j] = 1
        for i in range(j, n):
            sum = 0
            for k in range(j):
                sum += L[i][k] * U[k][j]
            L[i][j] = matrix[i][j] - sum
        for i in range(j + 1, n):
            sum = 0
            for k in range(j):
                sum += L[j][k] * U[k][i]
            U[j][i] = (matrix[j][i] - sum) / L[j][j]
    return L, U

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    matrix = np.array(data['matrix'])
    method = data['method']

    if method == 'eigen':
        eigenvalues, eigenvectors = eigenvalues_and_vectors(matrix)
        result = {
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.tolist()
        }
    elif method == 'lu':
        P, L, U = lu_decomposition(matrix)
        result = {
            'P': P.tolist(),
            'L': L.tolist(),
            'U': U.tolist()
        }
    elif method == 'cholesky':
        L, U = cholesky_decomposition(matrix)
        if L is not None:
            result = {
                'L': L.tolist(),
                'L.T': U.tolist()
            }
        else:
            result = 'Cholesky decomposition not applicable.'
    elif method == 'doolittle':
        L, U = doolittle_decomposition(matrix)
        result = {
            'L': L.tolist(),
            'U': U.tolist()
        }
    elif method == 'crout':
        L, U = crout_decomposition(matrix)
        result = {
            'L': L.tolist(),
            'U': U.tolist()
        }
    else:
        result = 'Invalid method.'

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
