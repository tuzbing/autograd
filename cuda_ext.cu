
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_ops.cuh"

namespace py = pybind11;

#define CUDA_CHECK(call)                                                       \
	do {																	   \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err__));               \
        }                                                                      \
	} while (0)

class CudaContext {
	private:
		cudaStream_t stream_ = nullptr;
		size_t max_buf_len_ = 0;
		int num_sms_ = 1;
		std::unordered_map<std::string, const void*> kernels_;

	public:
		CudaContext() {
			int dev = 0;
			CUDA_CHECK(cudaGetDevice(&dev));
			cudaDeviceProp props{};
			CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
			max_buf_len_ = props.totalGlobalMem;
			num_sms_ = std::max(1, props.multiProcessorCount);
			CUDA_CHECK(cudaStreamCreate(&stream_));

			kernels_ = {
	            {"fill_scalar_kernel2",                       (const void*)&fill_scalar_kernel2},
	            {"gather_forward_kernel",                     (const void*)&gather_forward_kernel},
	            {"gather_backward_kernel",                    (const void*)&gather_backward_kernel},
	            {"fused_linear_add_forward_kernel",           (const void*)&fused_linear_add_forward_kernel},
	            {"cuda_sgemm_tn_kernel",                      (const void*)&cuda_sgemm_tn_kernel},
	            {"cuda_sgemm_nt_kernel",                      (const void*)&cuda_sgemm_nt_kernel},
	            {"linear_add_dx_backward_kernel",             (const void*)&linear_add_dx_backward_kernel},
	            {"bias_grad_reduce_kernel",                   (const void*)&bias_grad_reduce_kernel},
	            {"cuda_sgemm_nn_kernel",                      (const void*)&cuda_sgemm_nn_kernel},
	            {"rmsnorm_forward_kernel",                    (const void*)&rmsnorm_forward_kernel},
	            {"fused_wgu_swiglu_forward_kernel",           (const void*)&fused_wgu_swiglu_forward_kernel},
	            {"swiglu_forward_kernel",                     (const void*)&swiglu_forward_kernel},
	            {"swiglu_backward_inplace_kernel",            (const void*)&swiglu_backward_inplace_kernel},
	            {"rmsnorm_backward_kernel",                   (const void*)&rmsnorm_backward_kernel},
	            {"rmsnorm_dgamma_kernel",                     (const void*)&rmsnorm_dgamma_kernel},
	            {"fused_rms_fa_qkv_kernel",                   (const void*)&fused_rms_fa_qkv_kernel},
	            {"rope_qk_inplace_kernel",                    (const void*)&rope_qk_inplace_kernel},
	            {"flash_attn_fwd_kernel",                     (const void*)&flash_attn_fwd_kernel},
	            {"flash_attn_bwd_prepare_kernel",             (const void*)&flash_attn_bwd_prepare_kernel},
	            {"flash_attn_bwd_kernel",                     (const void*)&flash_attn_bwd_kernel},
	            {"fused_linear_ce_logits_to_dlogits_kernel",  (const void*)&fused_linear_ce_logits_to_dlogits_kernel},
	            {"fused_linear_ce_reduce_kernel",             (const void*)&fused_linear_ce_reduce_kernel},
	            {"ce_tile_logits_to_dlogits_kernel",          (const void*)&ce_tile_logits_to_dlogits_kernel},
	            {"fp32_to_bf16_accum_kernel",                 (const void*)&fp32_to_bf16_accum_kernel},
	            {"add_forward_kernel",                        (const void*)&add_forward_kernel},
	            {"add_residual_backward_kernel",              (const void*)&add_residual_backward_kernel},
	            {"grad_norm_sq_kernel",                       (const void*)&grad_norm_sq_kernel},
	            {"adamw_step_kernel",                         (const void*)&adamw_step_kernel},
	            {"adamw8bit_step_kernel",                     (const void*)&adamw8bit_step_kernel},
	            {"mini_update_v_kernel",                      (const void*)&mini_update_v_kernel},
	            {"adamw_mini8bit_step_kernel",                (const void*)&adamw_mini8bit_step_kernel},
	            {"cuda_sgemm_nn_128_kernel",                  (const void*)&cuda_sgemm_nn_128_kernel},
	            {"cuda_sgemm_nt_128_kernel",                  (const void*)&cuda_sgemm_nt_128_kernel},
	            {"cuda_sgemm_tn_128_kernel",                  (const void*)&cuda_sgemm_tn_128_kernel},
	            {"fused_linear_add_128_kernel",               (const void*)&fused_linear_add_128_kernel},
	            {"fused_wgu_swiglu_128_kernel",               (const void*)&fused_wgu_swiglu_128_kernel},
	            {"fused_rms_fa_qkv_128_kernel",               (const void*)&fused_rms_fa_qkv_128_kernel},
	        };
	    }

        size_t max_buffer_length() const {
        	return max_buf_len_;
        }

        int num_sms() const {
        	return num_sms_;
        }

        uintptr_t alloc(size_t nbytes) {
        	void* ptr = nullptr;
        	CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(nbytes, 16)));
        	return reinterpret_cast<uintptr_t>(ptr);
        }

        void free(uintptr_t buffer_id) {
        	if (buffer_id) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer_id)));
        }

        void memset_zero(uintptr_t buffer_id, size_t offset, size_t nbytes) {
        	if (nbytes == 0) return;
    		CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<char*>(buffer_id) + offset, 0, nbytes, stream_));
        }

        void begin_pass() {}

        void end_pass() {
        	CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        void synchronize() {
        	CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        void dispatch(const std::string& name,
        			  const std::vector<uintptr_t>& buf_ids,
        			  const std::vector<size_t>& offsets,
        			  const std::vector<uint32_t>& u32_args,
        			  uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        			  uint32_t block_x, uint32_t smem_bytes = 0) {
        	if (grid_x == 0 || grid_y == 0 || grid_z == 0) return;
        	auto it = kernels_.find(name);
        	if (it == kernels_.end()) {
        		throw std::runtime_error("CUDA kernel not registered: " + name);
        	}

        	std::vector<void*> dev_ptrs(buf_ids.size());
    		std::vector<void*> kargs;
    		kargs.reserve(buf_ids.size() + u32_args.size());
	        for (size_t i = 0; i < buf_ids.size(); ++i) {
	            dev_ptrs[i] = buf_ids[i] ? (reinterpret_cast<char*>(buf_ids[i]) + offsets[i]) : nullptr;
	            kargs.push_back(&dev_ptrs[i]);
	        }
	        for (size_t j = 0; j < u32_args.size(); ++j) {
	            kargs.push_back(const_cast<uint32_t*>(&u32_args[j]));
	        }

	        CUDA_CHECK(cudaLaunchKernel(it->second, dim3(grid_x, grid_y, grid_z), dim3(block_x, 1, 1), kargs.data(), smem_bytes, stream_));
        }

        // CPU -> GPU显存
        void copy_bytes_from_host(uintptr_t buffer_id, size_t offset, py::buffer host_buf) {
        	py::buffer_info info = host_buf.request();
        	size_t nbytes = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
        	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(buffer_id) + offset, info.ptr, nbytes, cudaMemcpyHostToDevice, stream_));
        	CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        // GPU显存 -> CPU
        py::bytes copy_bytes_to_host(uintptr_t buffer_id, size_t offset, size_t nbytes) {
        	std::string out(nbytes, '\0');
        	CUDA_CHECK(cudaMemcpyAsync(out.data(), reinterpret_cast<const char*>(buffer_id) + offset, nbytes, cudaMemcpyDeviceToHost, stream_));
        	CUDA_CHECK(cudaStreamSynchronize(stream_));
        	return py::bytes(out);
        }

        void copy_from_host(uintptr_t buffer_id, size_t offset, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        	py::buffer_info info = arr.request();
        	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(buffer_id) + offset, info.ptr,
        							   static_cast<size_t>(info.size)*sizeof(float), cudaMemcpyHostToDevice, stream_));
        	CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        py::array_t<float> copy_to_host(uintptr_t buffer_id, size_t offset, py::tuple shape) {
	        std::vector<ssize_t> shape_vec;
	        size_t numel = 1;
	        for (auto d : shape) {
	            size_t dim = d.cast<size_t>();
	            shape_vec.push_back(static_cast<ssize_t>(dim));
	            numel *= dim;
	        }
	        py::array_t<float> result(shape_vec);
	        CUDA_CHECK(cudaMemcpyAsync(result.mutable_data(),
	                                   reinterpret_cast<const char*>(buffer_id) + offset,
	                                   numel * sizeof(float), cudaMemcpyDeviceToHost, stream_));
	        CUDA_CHECK(cudaStreamSynchronize(stream_));
	        return result;
	    }
};

static CudaContext& get_ctx() {
	static CudaContext ctx;
	return ctx;
}

PYBIND11_MODULE(cuda_ext, m) {
    py::class_<CudaContext>(m, "CudaContext")
        .def("max_buffer_length", &CudaContext::max_buffer_length)
        .def("num_sms", &CudaContext::num_sms)
        .def("alloc", &CudaContext::alloc)
        .def("free", &CudaContext::free)
        .def("memset_zero", &CudaContext::memset_zero)
        .def("begin_pass", &CudaContext::begin_pass)
        .def("end_pass", &CudaContext::end_pass)
        .def("synchronize", &CudaContext::synchronize)
        .def("dispatch", &CudaContext::dispatch,
             py::arg("name"), py::arg("buf_ids"), py::arg("offsets"), py::arg("u32_args"),
             py::arg("grid_x"), py::arg("grid_y"), py::arg("grid_z"), py::arg("block_x"),
             py::arg("smem_bytes") = 0)
        .def("copy_bytes_from_host", &CudaContext::copy_bytes_from_host)
        .def("copy_bytes_to_host", &CudaContext::copy_bytes_to_host)
        .def("copy_from_host", &CudaContext::copy_from_host)
        .def("copy_to_host", &CudaContext::copy_to_host);
    m.def("get_ctx", &get_ctx, py::return_value_policy::reference);
}