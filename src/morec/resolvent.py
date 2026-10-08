"""Shifted linear solves (omega*I - H) x = b for many right-hand sides.

Sparse LU for dimensions below `iter_threshold`, ILU-preconditioned GMRES (BiCGSTAB fallback)
above it.  Factorizations are cached per frequency (LRU, `max_cached` entries).
"""
import multiprocessing
import sys
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .lanczos import shifted_solve


def _ilu_preconditioner(A_csc, omega, H):
    n = A_csc.shape[0]
    try:
        ilu = spla.spilu(A_csc, drop_tol=1e-3, fill_factor=2, diag_pivot_thresh=0.0)
        return ilu, spla.LinearOperator((n, n), ilu.solve, dtype=complex)
    except Exception:
        d = omega - H.diagonal()
        d = 1.0 / np.where(np.abs(d) < 1e-10, omega, d)
        return None, spla.LinearOperator((n, n), lambda x: d * x, dtype=complex)


def _gmres_columns(H, omega, M, rhs, tol, n_threads=1):
    n = H.shape[0]
    A = spla.LinearOperator((n, n), lambda x: omega * x - H @ x, dtype=complex)
    restart = min(200, max(50, n // 500))
    maxiter = max(200, min(2 * restart, 1000))

    def one(i):
        b = rhs[:, i].copy()
        x, info = spla.gmres(A, b, M=M, rtol=tol, atol=tol * 1e-2, restart=restart, maxiter=maxiter)
        if info != 0:
            x, info2 = spla.bicgstab(A, b, M=M, rtol=tol, atol=tol * 1e-2, maxiter=maxiter)
            if info2 != 0:
                warnings.warn(f"iterative solve did not converge (omega={omega}, column {i})")
        return x

    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            return np.column_stack(list(pool.map(one, range(rhs.shape[1]))))
    return np.column_stack([one(i) for i in range(rhs.shape[1])])


class ResolventSolver:
    def __init__(self, H, max_cached=8, iter_threshold=50_000, n_workers=1):
        self.H = sp.csr_matrix(H)
        self.n = self.H.shape[0]
        self.max_cached = max_cached
        self.iter_threshold = int(iter_threshold)
        self.n_workers = max(1, int(n_workers))
        self._cache, self._order = {}, []

    @property
    def iterative(self):
        return self.n >= self.iter_threshold

    def _cached(self, kind, omega, build):
        key = (kind, complex(round(omega.real, 10), round(omega.imag, 10)))
        if key in self._cache:
            self._order.remove(key)
        else:
            if self.max_cached > 0 and len(self._cache) >= self.max_cached:
                del self._cache[self._order.pop(0)]
            self._cache[key] = build()
        self._order.append(key)
        return self._cache[key]

    def _shifted(self, omega):
        return (omega * sp.eye(self.n, format="csc", dtype=np.complex128) - self.H.tocsc()).tocsc()

    def solve(self, omega, rhs, tol=1e-10):
        """(omega*I - H)^-1 rhs for a dense (n, k) right-hand side."""
        rhs = np.asarray(rhs).reshape(self.n, -1)
        if not self.iterative:
            try:
                lu = self._cached("lu", omega, lambda: spla.splu(self._shifted(omega)))
                return np.column_stack([lu.solve(rhs[:, i]) for i in range(rhs.shape[1])])
            except RuntimeError:
                tol = 1e-10
        _, M = self._cached("ilu", omega,
                            lambda: _ilu_preconditioner(self._shifted(omega), omega, self.H))
        return _gmres_columns(self.H, omega, M, rhs, tol, self.n_workers)

    def solve_parallel(self, omegas, rhs, tol=1e-6, n_workers=None):
        """One (omega*I - H)^-1 rhs block per frequency, one frequency per worker process."""
        n_workers = self.n_workers if n_workers is None else n_workers
        per_worker = rhs.nbytes + rhs.size * 16
        try:
            import psutil
            n_workers = max(1, min(n_workers, int(psutil.virtual_memory().available
                                                  / (3 * max(per_worker, 1)))))
        except ImportError:
            pass
        H = self.H
        init = (H.data.copy(), H.indices.copy(), H.indptr.copy(), H.shape, tol, rhs, sys.path[:])
        with multiprocessing.get_context("spawn").Pool(min(n_workers, len(omegas)),
                                                       initializer=_worker_init, initargs=init) as pool:
            return list(pool.map(_worker_solve, list(omegas)))


_W = {}


def _worker_init(data, indices, indptr, shape, tol, rhs, path):
    for p in path:
        if p not in sys.path:
            sys.path.insert(0, p)
    _W.update(H=sp.csr_matrix((data, indices, indptr), shape=shape), tol=tol, rhs=rhs)


def _worker_solve(omega):
    H = _W["H"]
    A = (omega * sp.eye(H.shape[0], format="csc", dtype=np.complex128) - sp.csc_matrix(H)).tocsc()
    _, M = _ilu_preconditioner(A, omega, H)
    return _gmres_columns(H, omega, M, _W["rhs"], _W["tol"]).reshape(H.shape[0], -1)


class ShiftedLanczos:
    """(omega - A)^-1 rhs at all frequencies from one Lanczos run per right-hand side.

    No factorisation: the cost of a column is one Krylov space shared by every frequency, so
    it hardly depends on their number.  A must be real symmetric and rhs real.
    """
    iterative = False

    def __init__(self, A, tol=1e-6):
        self.A, self.tol = A.tocsr(), tol

    def solve_all(self, omega, rhs):
        omega = np.asarray(omega)
        X = shifted_solve(self.A, np.asarray(rhs).real, omega, plus=False, tol=self.tol)
        return [X[:, a::len(omega)] for a in range(len(omega))]
