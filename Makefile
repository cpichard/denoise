
# From rez env openexr-2
OPENEXR_INCLUDE_DIR = /home/cyril/opt/openexr-2.2.0/include/OpenEXR
ILMBASE_INCLUDE_DIR = /home/cyril/opt/ilmbase-2.2.0/include/OpenEXR
OPENEXR_LIBRARY_DIR = /home/cyril/opt/openexr-2.2.0/lib
ILMBASE_LIBRARY_DIR = /home/cyril/opt/ilmbase-2.2.0/lib

GCC ?= g++
CUDA_PATH ?= "/usr/local/cuda"
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)

denoise: main.o exrloader.o nlm.o
	$(NVCC) -g -o denoise main.o exrloader.o nlm.o -L$(OPENEXR_LIBRARY_DIR) -L$(ILMBASE_LIBRARY_DIR) -lHalf -lIex -lIexMath -lImath -lIlmThread -lIlmImf 

main.o: main.cpp
	$(NVCC) -g -I$(OPENEXR_INCLUDE_DIR) -I$(ILMBASE_INCLUDE_DIR) -c main.cpp

exrloader.o: exrloader.cpp
	$(NVCC) -g -I$(OPENEXR_INCLUDE_DIR) -I$(ILMBASE_INCLUDE_DIR) -c exrloader.cpp

nlm.o: nlm.cu nlm.hpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

clean:
	rm *.o denoise



