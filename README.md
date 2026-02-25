# N-Species MCTDHF

Multi-Configuration Time-Dependent Hartree for N distinguishable species using [JAX](https://github.com/jax-ml/jax).

The CI part of the calculation is "semi-direct", storing single- and two-particle overlaps but only calculating HC products instead of diagonalising the full Hamiltonian.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/leoskar/n-species-mctdh.git
```

With CUDA support:

```bash
pip install "mctdhf-n-species[cuda12] @ git+https://github.com/leoskar/n-species-mctdh.git"
```

## Dependencies

- JAX (≥ 0.4)
- NumPy (≥ 1.24)
- SciPy (≥ 1.10)
- opt-einsum (≥ 3.3)
- Matplotlib (≥ 3.5)

## License

MIT
