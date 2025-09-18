"""The base class of QAIA."""
# pylint: disable=invalid-name
import numpy as np
from scipy import sparse as sp
import scipy.sparse as scsp
from scipy.sparse import csr_matrix
from type_value_check import _check_int_type, _check_value_should_not_less, _check_number_type

try:
    import torch

    assert torch.cuda.is_available()
    _INSTALL_TORCH = True
except (ImportError, AssertionError):
    _INSTALL_TORCH = False


def brute_force_ground_state(J, h=None):
    """Brute-force search for ground state: returns (min_energy, best_spin_vector)."""
    N = J.shape[0]
    best_E = np.inf
    best_s = None
    # iterate all 2^N configurations
    for idx in range(1 << N):
        s = np.array([1 if (idx >> i) & 1 else -1 for i in range(N)])
        E = -0.5 * s @ J @ s
        if h is not None:
            E -= h.reshape(-1,) @ s
        if E < best_E:
            best_E = E
            best_s = s.copy()
    return best_E, best_s

class QAIA:
    r"""
    The base class of QAIA.

    This class contains the basic and common functions of all the algorithms.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N x 1).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        backend (str): Computation backend and precision to use: 'cpu-float32','gpu-float32',
            'gpu-float16', 'gpu-int8','npu-float32'. Default: ``'cpu-float32'``.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, backend='cpu-float32'):
        """Construct a QAIA algorithm."""
        valid_backends = {'cpu-float32', 'gpu-float32', 'gpu-float16', 'gpu-int8'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        if backend == "gpu-float32" and not _INSTALL_TORCH:
            raise ImportError("Please install pytorch before using qaia gpu backend, ensure environment has any GPU.")
        if not isinstance(J, (np.ndarray, sp.spmatrix)):
            raise TypeError(f"J requires numpy.array or scipy sparse matrix, but get {type(J)}")
        if len(J.shape) != 2 or J.shape[0] != J.shape[1]:
            raise ValueError(f"J must be a square matrix, but got shape {J.shape}")
        if isinstance(J, np.ndarray):
            if not np.allclose(J, J.T):
                raise ValueError("J must be a symmetric matrix.")
            if not np.all(np.diag(J) == 0):
                raise ValueError("The diagonal elements of J are not all 0, recommend transferring them to h.")
        if isinstance(J, sp.spmatrix):
            if (J != J.T).nnz != 0:
                raise ValueError("J must be a symmetric matrix.")
            if not np.all(J.diagonal() == 0):
                raise ValueError("The diagonal elements of J are not all 0, recommend transferring them to h.")

        if h is not None:
            if not isinstance(h, np.ndarray):
                raise TypeError(f"h requires numpy.array, but get {type(h)}")
            if h.shape != (J.shape[0],) and h.shape != (J.shape[0], 1):
                raise ValueError(f"h must have shape ({J.shape[0]},) or ({J.shape[0]}, 1), but got {h.shape}")
            if len(h.shape) == 1:
                h = h[:, np.newaxis]

        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError(f"x requires numpy.array, but get {type(x)}")
            if len(x.shape) != 2:
                raise ValueError(f"x must be a 2D array, but got shape {x.shape}")
            if x.shape[0] != J.shape[0] or x.shape[1] != batch_size:
                raise ValueError(f"x must have shape ({J.shape[0]}, {batch_size}), but got {x.shape}")

        _check_int_type("n_iter", n_iter)
        _check_value_should_not_less("n_iter", 1, n_iter)
        _check_int_type("batch_size", batch_size)
        _check_value_should_not_less("batch_size", 1, batch_size)

        if backend == "gpu-float32" and _INSTALL_TORCH:

            # If J is sparse, densify; otherwise use the ndarray directly
            if isinstance(J, scsp.spmatrix):
                dense_J = J.toarray()
            else:
                dense_J = J

            # Create a CUDA tensor
            J_tensor = torch.tensor(dense_J, dtype=torch.float32, device="cuda")

            # Convert to sparse CSR if possible (for memory/compute)
            try:
                J = J_tensor.to_sparse_csr()
            except RuntimeError:
                # If conversion fails, just keep the dense tensor
                J = J_tensor

            # Move h onto GPU as well
            if h is not None:
                h = torch.from_numpy(h).float().to("cuda")


        self.J = J
        self.h = h
        self.x = x
        # The number of spins
        self.N = self.J.shape[0]
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.backend = backend

    def initialize(self):
        """Randomly initialize spin values."""
        if self.x is None:
            if self.backend == "cpu-float32":
                self.x = 0.02 * (np.random.rand(self.N, self.batch_size) - 0.5)
            elif self.backend == "gpu-float32":
                self.x = 0.02 * (torch.rand(self.N, self.batch_size, device="cuda") - 0.5)

        else:
            if self.backend == "gpu-float32":
                self.x = torch.from_numpy(self.x).float().to("cuda")

    def calc_cut(self, x=None):
        r"""
        Calculate cut value.

        Args:
            x (numpy.array): The spin value with shape (N x batch_size).
                If ``None``, the initial spin will be used. Default: ``None``.
        """
        if self.backend in ["cpu-float32", 'gpu-float16', 'gpu-int8']:
            if x is None:
                sign = np.sign(self.x)
            else:
                sign = np.sign(x)
            return 0.25 * np.sum(self.J.dot(sign) * sign, axis=0) - 0.25 * self.J.sum()

        if self.backend == "gpu-float32":
            if x is None:
                sign = torch.sign(self.x)
            else:
                sign = torch.sign(x)
            return 0.25 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0) - 0.25 * self.J.sum()

        raise ValueError("invalid backend")

    def calc_energy(self, x=None):
        r"""
        Calculate energy.

        Args:
            x (numpy.array): The spin value with shape (N x batch_size).
                If ``None``, the initial spin will be used. Default: ``None``.
        """
        if self.backend in ["cpu-float32", 'gpu-float16', 'gpu-int8']:
            if x is None:
                sign = np.sign(self.x)
            else:
                sign = np.sign(x)

            if self.h is None:
                return -0.5 * np.sum(self.J.dot(sign) * sign, axis=0),sign
            return -0.5 * np.sum(self.J.dot(sign) * sign, axis=0, keepdims=True) - self.h.T @ sign, sign

        if self.backend == "gpu-float32":
            if x is None:
                sign = torch.sign(self.x)
            else:
                sign = torch.sign(x)

            if self.h is None:
                return -0.5 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0), sign
            return -0.5 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0, keepdim=True) - self.h.T @ sign, sign

        raise ValueError("invalid backend")


class OverflowException(Exception):
    r"""
    Custom exception class for handling overflow errors in numerical calculations.

    Args:
        message: Exception message string, defaults to "Overflow error".
    """

    def __init__(self, message="Overflow error"):
        self.message = message
        super().__init__(self.message)


"""Coherent Ising Machine with chaotic feedback control algorithm."""

class CFC(QAIA):
    r"""
    Coherent Ising Machine with chaotic feedback control algorithm.

    Reference: `Coherent Ising machines with optical error correction
    circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values drawn from normal distribution N(0, 0.1). Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``0.1``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float32'. Default: ``'cpu-float32'``.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, dt=0.1, backend='cpu-float32'):
        """Construct CFC algorithm."""
        _check_number_type("dt", dt)
        _check_value_should_not_less("dt", 0, dt)
        super().__init__(J, h, x, n_iter, batch_size, backend)
        if self.backend == "cpu-float32":
            self.J = csr_matrix(self.J)

        self.dt = dt
        # The number of first iterations
        self.Tr = int(0.9 * self.n_iter)
        # The number of additional iterations
        self.Tp = self.n_iter - self.Tr
        self.N = self.J.shape[0]
        # pumping parameters
        self.p = np.hstack([np.linspace(-1, 1, self.Tr), np.ones(self.Tp)])
        # target amplitude
        self.alpha = 1.0
        # coupling strength
        if self.backend == "cpu-float32":
            self.xi = np.sqrt(2 * self.N / np.sum(self.J**2))
        if self.backend == "gpu-float32":
            self.xi = torch.sqrt(2 * self.N / torch.sum(self.J.to_dense() ** 2))
        # rate of change of error variables
        self.beta = 0.15
        self.initialize()

    def initialize(self):
        """Initialize spin values and error variables."""
        if self.backend == "cpu-float32":
            if self.x is None:
                self.x = np.random.normal(0, 0.1, size=(self.N, self.batch_size))

            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

            self.e = np.ones_like(self.x)

        elif self.backend == "gpu-float32":
            if self.x is None:
                self.x = torch.normal(0, 0.1, size=(self.N, self.batch_size)).to("cuda")
            else:
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("cuda")

            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

            self.e = torch.ones_like(self.x, device="cuda")

    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution."""
        if self.backend == "cpu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    z = self.xi * self.e * (self.J @ self.x)
                    self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x + z) * self.dt
                    self.e = self.e + (-self.beta * self.e * (z**2 - self.alpha)) * self.dt

                    cond = np.abs(self.x) > 1.5
                    self.x = np.where(cond, 1.5 * np.sign(self.x), self.x)
                    self.e = np.where(self.e < 0.01, 0.01, self.e)
            else:
                for i in range(self.n_iter):
                    z = self.xi * self.e * (self.J @ self.x + self.h)
                    self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x + z) * self.dt
                    self.e = self.e + (-self.beta * self.e * (z**2 - self.alpha)) * self.dt

                    cond = np.abs(self.x) > 1.5
                    self.x = np.where(cond, 1.5 * np.sign(self.x), self.x)
                    self.e = np.where(self.e < 0.01, 0.01, self.e)

        elif self.backend == "gpu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    z = self.xi * self.e * (torch.sparse.mm(self.J, self.x))
                    self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x + z) * self.dt
                    self.e = self.e + (-self.beta * self.e * (z**2 - self.alpha)) * self.dt

                    cond = torch.abs(self.x) > 1.5
                    self.x = torch.where(cond, 1.5 * torch.sign(self.x), self.x)
                    self.e = torch.where(self.e < 0.01, 0.01, self.e)
            else:
                for i in range(self.n_iter):
                    z = self.xi * self.e * (torch.sparse.mm(self.J, self.x) + self.h)
                    self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x + z) * self.dt
                    self.e = self.e + (-self.beta * self.e * (z**2 - self.alpha)) * self.dt

                    cond = torch.abs(self.x) > 1.5
                    self.x = torch.where(cond, 1.5 * torch.sign(self.x), self.x)
                    self.e = torch.where(self.e < 0.01, 0.01, self.e)

# Benchmark block: generate random J and h and run CFC
if __name__ == '__main__':
    N = 10  # number of spins
    batch_size = 50
    # Generate random symmetric coupling matrix J with zero diagonal
    A = 10*np.random.randn(N, N)
    J = (A + A.T) / 2
    np.fill_diagonal(J, 0)

    # Generate random external field h
    h = 2*np.random.randn(N, 1)

    # Initialize and run CFC
    solver = CFC(-2*J, h=-h.reshape(N,), batch_size=batch_size, n_iter=1000, dt=0.4, backend='gpu-float32')
    solver.update()

    # Compute and display results
    # cuts = solver.calc_cut()
    energies, spins = solver.calc_energy()
    # print('Cut values:', cuts)
    print('Energy values:', energies)
    print('Spin configurations (signs):', spins)

    # Brute-force ground state
    bf_E, bf_s = brute_force_ground_state(-2*J, -h)
    print('Brute-force min energy:', bf_E)
    print('Brute-force spin:', bf_s)

"""Coherent Ising Machine with chaotic amplitude control algorithm."""
# pylint: disable=invalid-name

try:
    import torch

    assert torch.cuda.is_available()
    _INSTALL_TORCH = True
except (ImportError, AssertionError):
    _INSTALL_TORCH = False

class CAC(QAIA):
    r"""
    Coherent Ising Machine with chaotic amplitude control algorithm.

    Reference: `Coherent Ising machines with optical error correction
    circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values drawn from normal distribution N(0, 10^(-4)). Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``0.075``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float32','npu-float32'. Default: ``'cpu-float32'``.
     """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, dt=0.075, backend='cpu-float32'):
        """Construct CAC algorithm."""
        _check_number_type("dt", dt)
        _check_value_should_not_less("dt", 0, dt)
        super().__init__(J, h, x, n_iter, batch_size, backend)
        if self.backend == "cpu-float32":
            self.J = csr_matrix(self.J)

        self.N = self.J.shape[0]
        self.dt = dt
        # The number of first iterations
        self.Tr = int(0.9 * self.n_iter)
        # The number of additional iterations
        self.Tp = self.n_iter - self.Tr
        # pumping parameters
        self.p = np.hstack([np.linspace(-0.5, 1, self.Tr), np.ones(self.Tp)])
        # target amplitude
        self.alpha = np.hstack([np.linspace(1, 3, self.Tr), 3.0 * np.ones(self.Tp)])
        # coupling strength
        self.xi = None
        if self.backend == "cpu-float32":
            self.xi = np.sqrt(2 * self.N / np.sum(self.J**2))
        if self.backend == "gpu-float32":
            self.xi = torch.sqrt(2 * self.N / torch.sum(self.J.to_dense() ** 2))
        # rate of change of error variables
        self.beta = 0.3
        self.initialize()

    def initialize(self):
        """Initialize spin values and error variables."""
        if self.backend == "cpu-float32":
            if self.x is None:
                self.x = np.random.normal(0, 10 ** (-4), size=(self.N, self.batch_size))

            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

            self.e = np.ones((self.N, self.batch_size))

        elif self.backend == "gpu-float32":
            if self.x is None:
                self.x = torch.normal(0, 10 ** (-4), size=(self.N, self.batch_size)).to("cuda")
            else:
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("cuda")

            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

            self.e = torch.ones(self.N, self.batch_size, device="cuda")

        
    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution."""
        if self.backend == "cpu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (-self.x**3 + (self.p[i] - 1) * self.x + self.xi * self.e * (self.J @ self.x)) * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = np.abs(self.x) > (1.5 * np.sqrt(self.alpha[i]))
                    self.x = np.where(cond, 1.5 * np.sign(self.x) * np.sqrt(self.alpha[i]), self.x)
            else:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (-self.x**3 + (self.p[i] - 1) * self.x + self.xi * self.e * (self.J @ self.x + self.h))
                        * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = np.abs(self.x) > (1.5 * np.sqrt(self.alpha[i]))
                    self.x = np.where(cond, 1.5 * np.sign(self.x) * np.sqrt(self.alpha[i]), self.x)

        elif self.backend == "gpu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (-self.x**3 + (self.p[i] - 1) * self.x + self.xi * self.e * torch.sparse.mm(self.J, self.x))
                        * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = torch.abs(self.x) > (1.5 * torch.sqrt(torch.tensor(self.alpha[i])))
                    self.x = torch.where(
                        cond, 1.5 * torch.sign(self.x) * torch.sqrt(torch.tensor(self.alpha[i])), self.x
                    )
            else:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (
                            -self.x**3
                            + (self.p[i] - 1) * self.x
                            + self.xi * self.e * (torch.sparse.mm(self.J, self.x) + self.h)
                        )
                        * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = torch.abs(self.x) > (1.5 * torch.sqrt(torch.tensor(self.alpha[i])))
                    self.x = torch.where(
                        cond, 1.5 * torch.sign(self.x) * torch.sqrt(torch.tensor(self.alpha[i])), self.x
                    )


try:
    import torch

    assert torch.cuda.is_available()
    _INSTALL_TORCH = True
except (ImportError, AssertionError):
    _INSTALL_TORCH = False


class SFC(QAIA):
    r"""
    Coherent Ising Machine with separated feedback control algorithm.

    Reference: `Coherent Ising machines with optical error correction
    circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values drawn from normal distribution N(0, 0.1). Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``0.1``.
        k (float): parameter of deviation between mean-field and error variables. Default: ``0.2``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float32','npu-float32'. Default: ``'cpu-float32'``.

    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, dt=0.1, k=0.2, backend='cpu-float32'):
        """Construct SFC algorithm."""
        _check_number_type("dt", dt)
        _check_value_should_not_less("dt", 0, dt)

        _check_number_type("k", k)
        _check_value_should_not_less("k", 0, k)

        super().__init__(J, h, x, n_iter, batch_size, backend)
        if self.backend == "cpu-float32":
            self.J = csr_matrix(self.J)

        self.N = self.J.shape[0]
        self.dt = dt
        self.n_iter = n_iter
        self.k = k
        # pumping parameters
        self.p = np.linspace(-1, 1, self.n_iter)
        # coupling strength
        if self.backend == "cpu-float32":
            self.xi = np.sqrt(2 * self.N / np.sum(self.J**2))
        if self.backend == "gpu-float32":
            self.xi = torch.sqrt(2 * self.N / torch.sum(self.J.to_dense() ** 2))
        # rate of change of error variables
        self.beta = np.linspace(0.3, 0, self.n_iter)
        # coefficient of mean-field term
        self.c = np.linspace(1, 3, self.n_iter)
        self.initialize()

    def initialize(self):
        """Initialize spin values and error variables."""
        if self.backend == "cpu-float32":
            if self.x is None:
                self.x = np.random.normal(0, 0.1, (self.N, self.batch_size))
            self.e = np.zeros_like(self.x)

        elif self.backend == "gpu-float32":
            if self.x is None:
                self.x = torch.normal(0, 0.1, (self.N, self.batch_size)).to("cuda")
            else:
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("cuda")
            self.e = torch.zeros_like(self.x, device="cuda")

        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution."""
        if self.backend == "cpu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    z = -self.xi * (self.J @ self.x)
                    f = np.tanh(self.c[i] * z)
                    self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x - f - self.k * (z - self.e)) * self.dt
                    self.e = self.e + (-self.beta[i] * (self.e - z)) * self.dt

                    if np.isnan(self.x).any():
                        raise ValueError(
                            f"NaNs detected in tensor 'x'. "
                            f"This often indicates numerical instability. "
                            f"Consider adjusting parameters like dt={self.dt} or xi={self.xi}."
                        )
            else:
                for i in range(self.n_iter):
                    z = -self.xi * (self.J @ self.x + self.h)
                    f = np.tanh(self.c[i] * z)
                    self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x - f - self.k * (z - self.e)) * self.dt
                    self.e = self.e + (-self.beta[i] * (self.e - z)) * self.dt

                    if np.isnan(self.x).any():
                        raise ValueError(
                            f"NaNs detected in tensor 'x'. "
                            f"This often indicates numerical instability. "
                            f"Consider adjusting parameters like dt={self.dt} or xi={self.xi}."
                        )

        elif self.backend == "gpu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    z = -self.xi * (torch.sparse.mm(self.J, self.x))
                    f = torch.tanh(self.c[i] * z)
                    self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x - f - self.k * (z - self.e)) * self.dt
                    self.e = self.e + (-self.beta[i] * (self.e - z)) * self.dt

                    if torch.isnan(self.x).any():
                        raise ValueError(
                            f"NaNs detected in tensor 'x'. "
                            f"This often indicates numerical instability. "
                            f"Consider adjusting parameters like dt={self.dt} or xi={self.xi}."
                        )
            else:
                for i in range(self.n_iter):
                    z = -self.xi * (torch.sparse.mm(self.J, self.x) + self.h)
                    f = torch.tanh(self.c[i] * z)
                    self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x - f - self.k * (z - self.e)) * self.dt
                    self.e = self.e + (-self.beta[i] * (self.e - z)) * self.dt

                    if torch.isnan(self.x).any():
                        raise ValueError(
                            f"NaNs detected in tensor 'x'. "
                            f"This often indicates numerical instability. "
                            f"Consider adjusting parameters like dt={self.dt} or xi={self.xi}."
                        )

