"""
Sample code automatically generated on 2023-08-25 10:56:45

by geno from www.geno-project.org

from input

parameters
  matrix A
variables
  vector x
min
  -sum(tanh(A*x))+norm1(x)
st
  sum(x) >= 2
  sum(x) <= 10
  0 <= x
  1 >= x


Original problem has been transformed into

parameters
  matrix A
variables
  vector x
  vector tmp000
min
  sum(tmp000)-sum(tanh(A*x))
st
  2-sum(x) <= 0
  sum(x)-10 <= 0
  x-tmp000 <= vector(0)
  -(x+tmp000) <= vector(0)


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
try:
    from genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)



class GenoNLP:
    def __init__(self, A, np):
        self.np = np
        self.A = A
        assert isinstance(A, self.np.ndarray)
        dim = A.shape
        assert len(dim) == 2
        self.A_rows = dim[0]
        self.A_cols = dim[1]
        self.x_rows = self.A_cols
        self.x_cols = 1
        self.x_size = self.x_rows * self.x_cols
        self.tmp000_rows = self.A_cols
        self.tmp000_cols = 1
        self.tmp000_size = self.tmp000_rows * self.tmp000_cols
        # the following dim assertions need to hold for this problem
        assert self.tmp000_rows == self.x_rows == self.A_cols

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.x_size
        bounds += [0] * self.tmp000_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [1] * self.x_size
        bounds += [inf] * self.tmp000_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.xInit = self.np.zeros((self.x_rows, self.x_cols))
        self.tmp000Init = self.np.zeros((self.tmp000_rows, self.tmp000_cols))
        return self.np.hstack((self.xInit.reshape(-1), self.tmp000Init.reshape(-1)))

    def variables(self, _x):
        x = _x[0 : 0 + self.x_size]
        tmp000 = _x[0 + self.x_size : 0 + self.x_size + self.tmp000_size]
        return x, tmp000

    def fAndG(self, _x):
        x, tmp000 = self.variables(_x)
        t_0 = self.np.tanh((self.A).dot(x))
        f_ = (self.np.sum(tmp000) - self.np.sum(t_0))
        g_0 = -(self.A.T).dot((self.np.ones(self.A_rows) - (t_0 ** 2)))
        g_1 = self.np.ones(self.tmp000_rows)
        g_ = self.np.hstack((g_0, g_1))
        return f_, g_

    def functionValueIneqConstraint000(self, _x):
        x, tmp000 = self.variables(_x)
        f = (2 - self.np.sum(x))
        return f

    def gradientIneqConstraint000(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (-self.np.ones(self.x_rows))
        g_1 = (self.np.ones(self.tmp000_rows) * 0)
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint000(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = (-(_v * self.np.ones(self.x_rows)))
        gv_1 = (self.np.ones(self.tmp000_rows) * 0)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint001(self, _x):
        x, tmp000 = self.variables(_x)
        f = (self.np.sum(x) - 10)
        return f

    def gradientIneqConstraint001(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (self.np.ones(self.x_rows))
        g_1 = (self.np.ones(self.tmp000_rows) * 0)
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint001(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = ((_v * self.np.ones(self.x_rows)))
        gv_1 = (self.np.ones(self.tmp000_rows) * 0)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint002(self, _x):
        x, tmp000 = self.variables(_x)
        f = (x - tmp000)
        return f

    def gradientIneqConstraint002(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (self.np.eye(self.tmp000_rows, self.x_rows))
        g_1 = (-self.np.eye(self.tmp000_rows, self.tmp000_rows))
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint002(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = (_v)
        gv_1 = (-_v)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint003(self, _x):
        x, tmp000 = self.variables(_x)
        f = -(x + tmp000)
        return f

    def gradientIneqConstraint003(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (-self.np.eye(self.tmp000_rows, self.x_rows))
        g_1 = (-self.np.eye(self.tmp000_rows, self.tmp000_rows))
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint003(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = (-_v)
        gv_1 = (-_v)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

def solve(A, np):
    start = timer()
    NLP = GenoNLP(A, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    # These are the standard solver options, they can be omitted.
    options = {'eps_pg' : 1E-4,
               'constraint_tol' : 1E-4,
               'max_iter' : 3000,
               'm' : 10,
               'ls' : 0,
               'verbose' : 5  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.1.0')
        constraints = ({'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint000,
                        'jacprod' : NLP.jacProdIneqConstraint000},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint001,
                        'jacprod' : NLP.jacProdIneqConstraint001},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint002,
                        'jacprod' : NLP.jacProdIneqConstraint002},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint003,
                        'jacprod' : NLP.jacProdIneqConstraint003})
        result = minimize(NLP.fAndG, x0, lb=lb, ub=ub, options=options,
                      constraints=constraints, np=np)
    else:
        # SciPy: for inequality constraints need to change sign f(x) <= 0 -> f(x) >= 0
        constraints = ({'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint000(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint000(x)},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint001(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint001(x)},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint002(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint002(x)},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint003(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint003(x)})
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=list(zip(lb, ub)),
                          constraints=constraints)

    # assemble solution and map back to original problem
    x, tmp000 = NLP.variables(result.x)
    elapsed = timer() - start
    print('solving took %.3f sec' % elapsed)
    return result, x, tmp000

def generateRandomData(np):
    np.random.seed(0)
    A = np.random.randn(3, 3)
    return A

if __name__ == '__main__':
    import numpy as np
    # import cupy as np  # uncomment this for GPU usage
    print('\ngenerating random instance')
    A = generateRandomData(np=np)
    print('solving ...')
    result, x, tmp000 = solve(A, np=np)

