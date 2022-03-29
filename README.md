# Basic Sparse Matrix

Very minimal pure Rust sparse matrix lib for simple use. Not intented to replace any of the larger libs out there. Supported features:

Sparse matrices:
- Csr type
    - from existing data or sequentially entered data (increasing col by col, row by row only)
    - transpose
    - cholesky decomposition
    - density and NNZ reporting
    - element sum
    - multiplication against dense matrices
    - row access, both as compacted and filled results.

Dense matrix:
    - From raw data and sequentially entered data
    - column access

General
- linear system solver using cholesky decomposition

TODO:
- Sparse/Sparse multiplication
- Dense/Dense multiplication
- Addition of all kinds
- identity matrices.
- sparse element modification
- non-monotonic sparse element addition.
- Inplace operations to save memory/allocation time
- Eigendecomposition