# Local measurements — AlexPy v0.1 (preliminary)

These results are from an isolated development runtime, **not** a user-owned
machine or an independent benchmark service. They do **not** prove that AlexPy
is a faster Python interpreter. Results can fluctuate with CPU load.

- Interpreter: `3.13.5`
- Platform: `Linux-6.18.57-x86_64-with-glibc2.41`
- Architecture: `x86_64`
- Repetitions: 9, warmup: 3
- Baseline: `sum(x*x for x in values)`; competitor: `alexpy.sum_squares(values)`
- Native backend: `native-c`; input: plain `list(range(n))`

| Input length | CPython generator (median ms) | AlexPy C (median ms) | Relative acceleration | C path active |
|---:|---:|---:|---:|:---:|
| 1,000 | 0.0287 | 0.0036 | 8.04× | Yes |
| 100,000 | 3.0637 | 0.3535 | 8.67× | Yes |
| 250,000 | 7.6251 | 0.8653 | 8.81× | Yes |

Raw benchmark JSON is distributed separately and should be retained with any
portfolio claim. CPython 3.15 and multi-OS CI results **have not been verified**.
