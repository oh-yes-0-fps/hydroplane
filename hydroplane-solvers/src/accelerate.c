// Sparse factorizations through Accelerate, whose supernodal kernels run on Apple silicon's
// matrix units. A thin C layer so Rust never mirrors Accelerate's structs: one symbolic
// factorization (METIS ordering) per pattern, and an f32 and an f64 numeric factor refactored in
// place. `kind` is a `SparseFactorization_t` value: Cholesky and the LDLT variants take the lower
// triangle of a symmetric matrix, QR and LU the whole of a general one.
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    SparseMatrixStructure structure;
    SparseOpaqueSymbolicFactorization symbolic;
    SparseOpaqueFactorization_Float f32;
    SparseOpaqueFactorization_Double f64;
    int has_f32, has_f64;
    long *column_starts;
    int *row_indices;
} hs_accelerate;

// The symbolic factorization of an `rows x cols` pattern (CSC), or NULL with `*status` set.
void *hs_accelerate_new(int kind, long rows, long cols, int symmetric, const long *column_starts, const int *row_indices, int *status) {
    hs_accelerate *h = calloc(1, sizeof(hs_accelerate));
    long nnz = column_starts[cols];
    if (h) {
        h->column_starts = malloc(sizeof(long) * (cols + 1));
        h->row_indices = malloc(sizeof(int) * (nnz > 0 ? nnz : 1));
    }
    if (!h || !h->column_starts || !h->row_indices) {
        if (h) {
            free(h->column_starts);
            free(h->row_indices);
        }
        free(h);
        *status = (int)SparseInternalError;
        return NULL;
    }
    memcpy(h->column_starts, column_starts, sizeof(long) * (cols + 1));
    memcpy(h->row_indices, row_indices, sizeof(int) * nnz);
    SparseAttributes_t attributes = {.kind = SparseOrdinary};
    if (symmetric) {
        attributes.kind = SparseSymmetric;
        attributes.triangle = SparseLowerTriangle;
    }
    SparseMatrixStructure s = {
        .rowCount = (int)rows,
        .columnCount = (int)cols,
        .columnStarts = h->column_starts,
        .rowIndices = h->row_indices,
        .attributes = attributes,
        .blockSize = 1,
    };
    h->structure = s;
    // METIS rather than the default AMD: on 3D FEM tangents it leaves less fill and nearly halves
    // each refactorization.
    SparseSymbolicFactorOptions options = {
        .control = SparseDefaultControl,
        .orderMethod = SparseOrderMetis,
        .order = NULL,
        .ignoreRowsAndColumns = NULL,
        .malloc = malloc,
        .free = free,
        .reportError = NULL,
    };
    h->symbolic = SparseFactor((SparseFactorization_t)kind, s, options);
    *status = (int)h->symbolic.status;
    if (h->symbolic.status != SparseStatusOK) {
        SparseCleanup(h->symbolic);
        free(h->column_starts);
        free(h->row_indices);
        free(h);
        return NULL;
    }
    return h;
}

int hs_accelerate_factor_f32(void *p, float *values) {
    hs_accelerate *h = p;
    SparseMatrix_Float a = {.structure = h->structure, .data = values};
    if (h->has_f32) {
        SparseRefactor(a, &h->f32);
    } else {
        h->f32 = SparseFactor(h->symbolic, a);
        h->has_f32 = 1;
    }
    return (int)h->f32.status;
}

int hs_accelerate_factor_f64(void *p, double *values) {
    hs_accelerate *h = p;
    SparseMatrix_Double a = {.structure = h->structure, .data = values};
    if (h->has_f64) {
        SparseRefactor(a, &h->f64);
    } else {
        h->f64 = SparseFactor(h->symbolic, a);
        h->has_f64 = 1;
    }
    return (int)h->f64.status;
}

// x <- A^-1 x for a square matrix.
void hs_accelerate_solve_f32(void *p, float *xb) {
    hs_accelerate *h = p;
    DenseVector_Float v = {.count = h->structure.rowCount, .data = xb};
    SparseSolve(h->f32, v);
}

void hs_accelerate_solve_f64(void *p, double *xb) {
    hs_accelerate *h = p;
    DenseVector_Double v = {.count = h->structure.rowCount, .data = xb};
    SparseSolve(h->f64, v);
}

// x = argmin |A x - b| (least squares, rows >= cols) or the least-norm solution of A x = b
// (rows < cols), from a QR factorization.
void hs_accelerate_solve_rect_f32(void *p, float *b, float *x) {
    hs_accelerate *h = p;
    DenseVector_Float vb = {.count = h->structure.rowCount, .data = b};
    DenseVector_Float vx = {.count = h->structure.columnCount, .data = x};
    SparseSolve(h->f32, vb, vx);
}

void hs_accelerate_solve_rect_f64(void *p, double *b, double *x) {
    hs_accelerate *h = p;
    DenseVector_Double vb = {.count = h->structure.rowCount, .data = b};
    DenseVector_Double vx = {.count = h->structure.columnCount, .data = x};
    SparseSolve(h->f64, vb, vx);
}

// The pivots' signs of an LDLT factorization with threshold partial pivoting; non-zero on error.
int hs_accelerate_inertia_f32(void *p, int *positive, int *zero, int *negative) {
    hs_accelerate *h = p;
    return SparseGetInertia(h->f32, positive, zero, negative);
}

int hs_accelerate_inertia_f64(void *p, int *positive, int *zero, int *negative) {
    hs_accelerate *h = p;
    return SparseGetInertia(h->f64, positive, zero, negative);
}

void hs_accelerate_free(void *p) {
    hs_accelerate *h = p;
    if (h->has_f32) SparseCleanup(h->f32);
    if (h->has_f64) SparseCleanup(h->f64);
    SparseCleanup(h->symbolic);
    free(h->column_starts);
    free(h->row_indices);
    free(h);
}
