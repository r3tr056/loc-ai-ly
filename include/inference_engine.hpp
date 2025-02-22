#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <iostream>
#include <vector>
#include <cublas.h>
#include <cublas_v2.h>
#include <memory>
#include <kernels.hpp>
#include <cuda_bf16.h>


// GPU memory managerment
struct GPUData {
	float* data;
	size_t size;

	GPUData(size_t elements) : size(elements) { cudaMalloc(&data, elements * sizeof(float)); }
	~GPUData() { cudaFree(data); }
};

// GPU tensor
struct Tensor {
	std::vector<int> shape;
	GPUData gpu_data;
	ggml_type type;

	Tensor(const std::vector<int>& dims, ggml_type _type = GGML_TYPE_F32) : shape(dims), gpu_data(num_elements()), type(_type) {}
	Tensor(const std::vector<int>& dims, float* external_data, ggml_type _type = GGML_TYPE_F32) : shape(dims), gpu_data(0), type(_type) {
		gpu_data.data = external_data;
	}

	size_t num_elements() const {
		size_t count = 1;
		for (int dim : shape) count *= dim;
		return count;
	}
};

// Matrix Operations (BLAS)
struct Matrix {
	GPUData gpu_data;
	int rows, cols;
	cublasHandle_t cublas_handle;
	bool owns_data;
	ggml_type type;

	Matrix(int r, int c, cublasHandle_t handle, ggml_type _type = GGML_TYPE_F32) : rows(r), cols(c), cublas_handle(handle), gpu_data(r * c * get_type_size(_type)), owns_data(true), type(_type) {
		CHECK_CUDA(cudaMemset(gpu_data.data, 0, r * c * get_type_size(_type)));
	}

	Matrix(const std::vector<int>& dims, cublasHandle_t handle, ggml_type _type = GGML_TYPE_F32) : rows(dims[0]), cols(dims[1]), cublas_handle(handle), gpu_data(num_elements() * get_type_size(_type)), owns_data(true), type(_type) {}

	Matrix(const std::vector<int>& dims, float* external_data, cublasHandle_t handle, bool _owns_data = false, ggml_type _type = GGML_TYPE_F32) : rows(dims[0]), cols(dims[1]), cublas_handle(handle), gpu_data(0), owns_data(_owns_data), type(_type) {
		if (dims.size() != 2) {
			throw std::runtime_error("Matrix constructor with external data expected 2 dims.");
		}
		gpu_data.data = (float*)external_data;
	}

	~Matrix() {
		if (!owns_data) {
			gpu_data.data = nullptr;
			gpu_data.size = 0;
		}
	}

	float* data() {return gpu_data.data;}

	void* data_ptr() const { return gpu_data.data; }
	float* data_f32() const { return (float*)gpu_data.data; }
	half* data_f16() const { return (half*)gpu_data.data; }

	size_t num_elements() const { return rows * cols; }

	size_t get_type_size(ggml_type type) const {
		switch (type) {
			case GGML_TYPE_F16: return sizeof(half);
			case GGML_TYPE_BF16: return sizeof(__bf16);
			case GGML_TYPE_F32:
			default: return sizeof(float);
		}
	}

	void gemm(const Matrix& a, const Matrix& b, bool transpose_a = false, bool transpose_b = false) {
		cublasOperation_t transA = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t transB = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
		
		int m = rows;
		int n = cols;
		int k = a.cols;

		cudaDataType_t data_type = CUDA_R_32F;
		cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
		void *alpha_ptr, *beta_ptr;

		float alpha_f32 = 1.0f;
		float beta_f32 = 0.0f;

		half alpha_f16 = __float2half(1.0f);
		half beta_f16 = __float2half(0.0f);

		__bf16 alpha_bf16 = __float2bfloat16(1.0f);
		__bf16 beta_bf16 = __float2bfloat16(0.0f);

		if (type == GGML_TYPE_F16 || a.type == GGML_TYPE_F16 || b.type == GGML_TYPE_F16) {
			data_type == CUDA_R_16F;
			compute_type = CUBLAS_COMPUTE_16F;
			alpha_ptr = &alpha_f16;
			beta_ptr = &beta_f16;
		} else if (type == GGML_TYPE_BF16 || a.type == GGML_TYPE_BF16 || b.type == GGML_TYPE_BF16) {
			data_type = CUDA_R_16BF;
			compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
			alpha_ptr = &alpha_bf16;
			beta_ptr = &beta_bf16;
		}
		
		cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;

		cublasGemmEx(
			cublas_handle,
			transB, transA,
			n, m, k,
			alpha_ptr,
			b.data_ptr(), data_type, b.cols,
			a.data_ptr(), data_type, a.cols,
			beta_ptr,
			this->data_ptr(), data_type, cols,
			compute_type,
			algo
		);
	}

	void gemm_f16(const Matrix& a, const Matrix& b, bool transpose_a, bool transpose_b) {
		// --- FP16 GEMM using cublasHgemm ---
		const half alpha = __float2half(1.0f);
		const half beta  = __float2half(0.0f);
		cublasOperation_t transA = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t transB = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHgemm(
			cublas_handle,
			transB, transA,
			cols, rows, a.cols,
			&alpha,
			b.data_f16(), b.cols, // Use data_f16()
			a.data_f16(), a.cols, // Use data_f16()
			&beta,
			data_f16(), cols     // Use data_f16()
		);
	}

	void gemm_bf16(const Matrix& a, const Matrix& b, bool transpose_a = false, bool transpose_b = false) {
		// -- BF16 GEMM using cublasHgemm
		const __half alpha = __float2half(1.0f);
		const __half beta = __float2half(0.0f);

		cublasOperation_t transA = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t transB = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHgemm(
			cublas_handle,
			transB, transA,
			cols, rows, a.cols,
			&alpha,
			b.data_f16(), b.cols,
			a.data_f16(), a.cols,
			&beta,
			data_f16(), cols
		);
	}
};

struct Config {
    virtual ~Config() = default;
    virtual std::string model_type() const = 0;
};

class Model {
public:
    Model() = default;
    virtual ~Model() = default;

    virtual void forward(const std::vector<int>& tokens, Matrix& logits) = 0;

protected:
    std::unique_ptr<Config> config_;

private:
    Model(const Model&) = delete;        // Non-copyable
    Model& operator=(const Model&) = delete; // Non-copyable
    Model(Model&&) = delete;             // Non-movable (optional, but good practice for resource managers)
    Model& operator=(Model&&) = delete;  // Non-movable (optional)
};

#endif // INFERENCE_ENGINE_H