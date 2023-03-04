This library was created by Vaishnavi Gujjula, during the course of her Ph.D.

DAFMM2Dlib (Directional Algebraic Fast Multipole Method)is a library for computing fast matrix-vector products when the underlying kernel is oscillatory. It differs from the usual FMM, in the sense that the far-field region of boxes is divided into conical regions. The sub-matrices corresponding to the interactions in these conical regions are of low rank. The low rank approximations of appropriate matrix sub-blocks are formed in an algebraic fashion using ACA. The algorithm has been parallelized using OpenMP. The code is written in C++ and features an easy-to-use interface.

For more details on the usage of the library, visit the [documentation](http://dafmm2d.rtfd.io) page.
