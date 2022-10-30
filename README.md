# strassenMM
Efficient Strassen matrix multiplication and divide & conquer version matrix multiplication.
This is the assignment for my algorithm lesson during my Ph.D.

This implementation can process matrix in any size (not need to be pow of 2).
Multi-thread is not used,
but it is still faster than other implementations in github,
and costs less memory.

Note: the algorithms are implemented via C under **Linux**.
Please run
```bash
gcc main.c -o main -O3
```
under Linux to compile the C source files (-O3 is optional).
