#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

const size_t BLOCK_SIZE = 16;
const int PRECISION = 3;
const int N_MAX = 1024;
const int M_MAX = 9;

const std::string INPUT_FILE = "input.txt";
const std::string OUTPUT_FILE = "output.txt";

struct matrix_t {
    size_t size;
    std::vector<float> elements;

    explicit matrix_t()
            : size(0), elements(std::vector<float>()) {}

    explicit matrix_t(size_t size)
            : size(size), elements(std::vector<float>(size * size, 0)) {}

    size_t elements_num() {
        return elements.size();
    }
};

void read_input(std::string const &file, matrix_t &A, matrix_t &B) {
    std::ifstream input(file.c_str());
    size_t N, M;
    input >> N >> M;
    std::cout << N << " " << M << '\n';

    A = matrix_t(N);
    B = matrix_t(M);

    for (int i = 0; i < N * N; i++) {
        input >> A.elements[i];
    }

    for (int i = 0; i < M * M; i++) {
        input >> B.elements[i];
    }
    input.close();
}

void write_result(std::string const &file, matrix_t const &result) {
    std::ofstream output(file.c_str());
    size_t N = result.size;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            output << result.elements[i * N + j] << " ";
            std::cout << std::fixed << std::setprecision(PRECISION) << result.elements[i * N + j] << " ";
        }
        output << std::endl;
        std::cout << '\n';
    }
    output.close();
}

size_t calculate_thread_size(size_t N) {
    size_t thread_size = BLOCK_SIZE;
    while (thread_size < N) {
        thread_size *= 2;
    }
    return thread_size;
}

int main() {

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        } catch (cl::Error const &e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        matrix_t A, B;
        read_input(INPUT_FILE, A, B);
        /////////////////////////////////////////////////////////////
        for (int i = 0; i < A.elements.size(); i++) {
            std::cout << A.elements[i] << " ";
        }
        std::cout << '\n';
        for (int i = 0; i < B.elements.size(); i++) {
            std::cout << B.elements[i] << " ";
        }
        std::cout << '\n';
        /////////////////////////////////////////////////////////////
        matrix_t C(A.size);

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * N_MAX * N_MAX);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * M_MAX * M_MAX);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * N_MAX * N_MAX);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * A.elements_num(), A.elements.data());
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * B.elements_num(), B.elements.data());

        size_t thread_size = calculate_thread_size(A.size);
        std::cout << "thread size: " << thread_size << '\n';

        // load named kernel from opencl source
        queue.finish();
        cl::Kernel kernel_gmem(program, "gpu_convolution");
        cl::KernelFunctor convolution_gmem(
                kernel_gmem,
                queue,
                cl::NullRange,
                cl::NDRange(thread_size, thread_size),
                cl::NDRange(BLOCK_SIZE, BLOCK_SIZE)
        );

        cl::Event event = convolution_gmem(dev_a, dev_b, dev_c, cl_int(A.size), cl_int(B.size));
        event.wait();

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * A.elements_num(), C.elements.data());

        write_result(OUTPUT_FILE, C);

    } catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}