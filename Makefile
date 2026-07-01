CUDA_ARCH := sm_86

NVCC := nvcc

TARGET := main
PROF_TARGET := profile

SRC := main.cu \
       src/cpu_gemm.cu \
       src/naive_gemm.cu \
       src/tiled_gemm.cu \
	   src/thread_tiled_gemm.cu \
	   src/thread_tiled_gemm_8.cu \
	   src/thread_tiled_gemm_16.cu


PROF_SRC := profile.cu \
            src/naive_gemm.cu \
            src/tiled_gemm.cu \
			src/thread_tiled_gemm.cu \
			src/thread_tiled_gemm_8.cu \
			src/thread_tiled_gemm_16.cu

NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -Iinclude

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(PROF_TARGET): $(PROF_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

profile-run: $(PROF_TARGET)
	./$(PROF_TARGET)

profile-naive: $(PROF_TARGET)
	ncu --set full --kernel-name matmul_gpu_naive -o naive_report ./$(PROF_TARGET)

profile-tiled: $(PROF_TARGET)
	ncu --set full --kernel-name matmul_gpu_tiled -o tiled_report ./$(PROF_TARGET)

clean:
	rm -f $(TARGET) $(PROF_TARGET) naive_report.ncu-rep tiled_report.ncu-rep

.PHONY: all run clean profile-run profile-naive profile-tiled