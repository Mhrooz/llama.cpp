#ifndef GGML_RKNN_H
#define GGML_RKNN_H

#include "ggml.h"
#include "ggml-backend.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"



#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
// static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name) ;
// GGML_BACKEND_API ggml_backend_t ggml_backend_rknn_init(void);
// GGML_BACKEND_API bool ggml_backend_is_rknn(ggml_backend_t backend);
// GGML_BACKEND_API void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads);

// GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rknn_reg(void);

// void copy_submatrix_A_process(int A_row_block, const int A_block_row_cnt, int &A_row_ele_cnt, int64_t m, const int A_block_row, bool &is_last, int A_column_block, const int A_block_column_cnt, int &A_column_ele_cnt, int64_t ori_k, const int A_block_column, void *sub_A_data, void *A_block_start, int A_row_cnt, double &memcpy_duration);

// void copy_submatrix_B_process(bool &is_last, int B_row_block, const int B_block_row_cnt, int &B_row_ele_cnt, int64_t ori_k, const int B_block_row, int B_column_block, const int B_block_column_cnt, int &B_column_ele_cnt, int64_t sub_n, const int B_block_column, void *sub_B_data, void *B_block_start, double &memcpy_duration);

// void check_A_B_data(int64_t m, int64_t k, const void *A_data, int64_t n, void *B_data);


// void check_A00xB00_CPU(const int64_t A_row_00, const int64_t B_col_00, const int64_t A_col_00, void *sub_A_data, void *sub_B_data, float *dst, int64_t n);

// void *pad_sub_matrix(const int64_t A_row_01, const int64_t A_col_01, const int64_t A_col_00, const void *A_data, int64_t k);

// void check_pad(const int64_t row, const int64_t col, void *pad_A01);


//
// backend API
//

static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name) ;
GGML_BACKEND_API ggml_backend_t ggml_backend_rknn_init(void);
GGML_BACKEND_API bool ggml_backend_is_rknn(ggml_backend_t backend);
GGML_BACKEND_API void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rknn_reg(void);

void copy_submatrix_A_process(int A_row_block, const int A_block_row_cnt, int &A_row_ele_cnt, int64_t m, const int A_block_row, bool &is_last, int A_column_block, const int A_block_column_cnt, int &A_column_ele_cnt, int64_t ori_k, const int A_block_column, void *sub_A_data, void *A_block_start, int A_row_cnt, double &memcpy_duration);

void copy_submatrix_B_process(bool &is_last, int B_row_block, const int B_block_row_cnt, int &B_row_ele_cnt, int64_t ori_k, const int B_block_row, int B_column_block, const int B_block_column_cnt, int &B_column_ele_cnt, int64_t sub_n, const int B_block_column, void *sub_B_data, void *B_block_start, double &memcpy_duration);

void check_A_B_data(int64_t m, int64_t k, const void *A_data, int64_t n, void *B_data);


void check_A00xB00_CPU(const int64_t A_row_00, const int64_t B_col_00, const int64_t A_col_00, void *sub_A_data, void *sub_B_data, float *dst, int64_t n);

void *pad_sub_matrix(const int64_t A_row_01, const int64_t A_col_01, const int64_t A_col_00, const void *A_data, int64_t k, bool is_A);

void check_pad(const int64_t row, const int64_t col, void *pad_A01);

#ifdef  __cplusplus
}
#endif

#endif // GGML_RKNN_H

