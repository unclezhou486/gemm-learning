CUDA_ARCH := sm_86

NVCC := nvcc

TARGET := main
SRC := test.cu

NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)