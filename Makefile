EIGEN3_INCLUDE = /usr/local/include/eigen3/
CAFFE2_LIB = /usr/local/lib/

CAFFE2_LINKFLAGS = -L${CAFFE2_LIB} -lCaffe2_CPU -lglog -lprotobuf -lgflags

TARGET = intro

SOURCE_FILES = src/intro.cc

CXXFLAGS = -std=c++11 -I${EIGEN3_INCLUDE} 

all: ${TARGET}

${TARGET} : src/intro.o
	$(CXX) $< $(CAFFE2_LINKFLAGS) -o $@
clean:
	rm src/*.o 
	rm ${TARGET}

src/intro.o : src/intro.cc
	$(CXX) src/intro.cc $(CXXFLAGS)  -c -o src/intro.o

