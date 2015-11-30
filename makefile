GCC=nvcc

all: Multi-Pi

Multi-Pi:      Multi-Pi.cu
	$(GCC) -o Multi-Pi Multi-Pi.cu -lm
clean:
	$(RM)  Multi-Pi Multi-Pi.o
