
#pragma once 
#include <cuda_runtime.h>
#include <sstream>
#include <vector>

class cuda_timer {
public:
    cuda_timer() {
        cudaEventCreate(&m_tic);
        cudaEventCreate(&m_toc);
    }

    ~cuda_timer() {
        cudaEventDestroy(m_tic);
        cudaEventDestroy(m_toc);
    }

    void tic() {
        cudaEventRecord(m_tic, 0);
    }

    void toc() {
        cudaEventRecord(m_toc, 0);
        cudaEventSynchronize(m_toc);
    }

    float get_time() {
        float time;
        cudaEventElapsedTime(&time, m_tic, m_toc);
        return time;
    }

private:
    cudaEvent_t m_tic, m_toc;
};

// src: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define GC(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }

}


template<typename T>
class cuda_type {
public:
    cuda_type(T &variable):h_var(variable) {
        GC(cudaMalloc(&d_var, sizeof(T)));
        GC(cudaMemcpy(d_var, &h_var, sizeof(T), cudaMemcpyHostToDevice));
    }

    ~cuda_type() {
        cudaFree(d_var);
    }

    T* get_d() {
        return d_var;
    }

    T& get_h() {
        return h_var;
    }

    int mem_copy_back() {
        cudaError_t code = cudaMemcpy(&h_var, d_var, sizeof(T), cudaMemcpyDeviceToHost);
        GC(code);
        return code == cudaSuccess;
    }
private:
    T *d_var;
    T &h_var;
};

template<typename T>
class cuda_container {
public:
    cuda_container(std::vector<T> &ori):h_ori(ori) {
        size = sizeof(T) * h_ori.size();
        GC(cudaMalloc(&d_ori, size));
        GC(cudaMemcpy(d_ori, h_ori.data(), size, cudaMemcpyHostToDevice));
    }
    ~cuda_container() {
        cudaFree(d_ori);
    }

    T* get_d() {
        return d_ori;
    }

    int get_n() {
        return (int)h_ori.size();
    }

    std::vector<T>& get_h() {
		return h_ori;
	} 

    int mem_copy_back() {
        cudaError_t code = cudaMemcpy(h_ori.data(), d_ori, size, cudaMemcpyDeviceToHost);
        GC(code);
        return code == cudaSuccess;
    }

private:
    std::vector<T> &h_ori;
    T *d_ori;
    size_t size;
};
