import requests

from ..base.module import BaseANN
from .._surreal_common import SurrealBatchMixin, start_server, stop_server

class SurrealDiskAnn(SurrealBatchMixin, BaseANN):

    def __init__(self, metric, method_param):
        if metric == "euclidean":
            self._metric = 'EUCLIDEAN'
        elif metric == 'angular':
            self._metric = 'COSINE'
        else:
            raise RuntimeError(f"unknown metric {metric}")
        self._degree = method_param['degree']
        self._l_build = method_param['l_build']
        self._alpha = method_param['alpha']
        # Defensive cleanup in case a prior algorithm definition left a server
        # alive, then start fresh and block until /health is 200.
        stop_server()
        self._proc = start_server()
        self._session = requests.Session()
        self._session.auth = ('ann', 'ann')
        headers={
            "surreal-ns": 'main',
            "surreal-db": 'main',
            "Accept": "application/json",
        }
        self._session.headers.update(headers)

    def _sql(self, q):
        r = self._session.post('http://127.0.0.1:8000/sql', q)
        if r.status_code != 200:
            raise RuntimeError(f"{r.text}")
        return r

    def _create_index(self, dim):
        s = (
            f"DEFINE INDEX ix ON items FIELDS r DISKANN "
            f"DIMENSION {dim} DIST {self._metric} TYPE F32 "
            f"DEGREE {self._degree} L_BUILD {self._l_build} ALPHA {self._alpha}"
        )
        self._checked_sql(s)


    def _ingest(self, dim, X):
        # Fit the database per batch
        print("Ingesting vectors...")
        batch = max(20000 // dim, 1)
        q = ""
        l = 0
        t = 0
        for i, embedding in enumerate(X):
            v = embedding.tolist()
            l += 1
            q += f"CREATE items:{i} SET r={v} RETURN NONE;"
            if l == batch:
                self._checked_sql(q)
                q = ''
                t += l
                l = 0
                print(f"\r{t} vectors ingested", end = '')
        if l > 0:
            self._checked_sql(q)
            t += l
            print(f"\r{t} vectors ingested", end = '')

    def fit(self, X):
        dim = X.shape[1]
        self._create_index(dim)
        self._ingest(dim, X)
        print("\nIndex construction done")

    def _checked_sql(self, q):
        res = self._sql(q).json()
        for r in res:
            if r['status'] != 'OK':
                raise RuntimeError(f"Error: {r}")
        return res

    def set_query_arguments(self, l_search):
        self._ls = l_search
        print("L = " + str(self._ls))

    def _build_query_sql(self, v, n):
        # `v` already a Python list (the mixin pre-converts via .tolist()).
        return f"SELECT id FROM items WHERE r <|{n},{self._ls}|> {v};"

    def query(self, v, n):
        v = v.tolist()
        j = self._checked_sql(self._build_query_sql(v, n))
        items = []
        for item in j[0]['result']:
            id = item['id']
            items.append(int(id[6:]))
        return items

    def __str__(self):
        return f"SurrealDiskAnn(degree={self._degree}, l_build={self._l_build}, alpha={self._alpha}, l_search={self._ls})"

    def done(self) -> None:
        self._session.close()
        stop_server()
