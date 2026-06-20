CUDA_ARCH := sm_86

NVCC := nvcc

TARGET := main

SRC := main.cu \
       src/cpu_gemm.cu \
       src/naive_gemm.cu \
       src/tiled_gemm.cu

NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -Iinclude

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean