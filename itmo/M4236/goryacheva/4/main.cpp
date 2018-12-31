#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

const size_t BLOCK_SIZE = 256;
const int PRECISION = 3;
const int N_MAX = 1048576;

const std::string INPUT_FILE = "input.txt";
const std::string OUTPUT_FILE = "output.txt";


void read_input(std::string const &file, std::vector<float> &vec) {
    std::ifstream input(file.c_str());
    size_t N;
    input >> N;
    vec = std::vector<float>(N);
    for (int i = 0; i < N; i++) {
        input >> vec[i];
    }
    input.close();
}

void write_result(std::string const &file, std::vector<float> const &result, size_t size) {
    std::ofstream output(file.c_str());
    for (size_t i = 0; i < size; ++i) {
        output << std::fixed << std::setprecision(PRECISION) << result[i] << " ";
    }
    output << std::endl;
    output.close();
}

void fill_zeros(std::vector<float> &input) {
    while (input.size() % BLOCK_SIZE != 0) {
        input.push_back(0);
    }
}

std::vector<float> prefix_sum(std::vector<float> const &input_data) {

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
        std::ifstream cl_file("prefix.cl");
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
            exit(1);
        }

        std::vector<float> input = input_data;
        std::vector<float> result(input_data.size());

        // allocate device buffer to hold message
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
        cl::Buffer dev_res(context, CL_MEM_WRITE_ONLY, sizeof(float) * result.size());

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * input.size(), input.data());
        queue.finish();

        // load named kernel from opencl source
        cl::Kernel kernel_prefix(program, "gpu_prefix");
        cl::KernelFunctor prefix_gmem(
                kernel_prefix,
                queue,
                cl::NullRange,
                cl::NDRange(input.size()),
                cl::NDRange(BLOCK_SIZE)
        );

        cl::Event event = prefix_gmem(
                dev_input,
                dev_res,
                cl::__local(sizeof(float) * BLOCK_SIZE),
                cl::__local(sizeof(float) * BLOCK_SIZE)
        );

        event.wait();
        queue.enqueueReadBuffer(dev_res, CL_TRUE, 0, sizeof(float) * result.size(), result.data());

        if (input.size() <= BLOCK_SIZE) {
            return result;
        }

        size_t group_num = input.size() / BLOCK_SIZE;
        std::vector<float> group_sums;
        for (size_t i = 0; i < group_num; ++i) {
            group_sums.push_back(result[(i + 1) * BLOCK_SIZE - 1]);
        }

        fill_zeros(group_sums);
        std::vector<float> sums = prefix_sum(group_sums);

        cl::Buffer dev_sum(context, CL_MEM_READ_ONLY, sizeof(float) * sums.size());

        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * result.size(), result.data());
        queue.enqueueWriteBuffer(dev_sum, CL_TRUE, 0, sizeof(float) * sums.size(), sums.data());
        queue.finish();

        cl::Kernel kernel_merge(program, "gpu_merge");
        cl::KernelFunctor merge_sums(
                kernel_merge,
                queue,
                cl::NullRange,
                cl::NDRange(input.size()),
                cl::NDRange(BLOCK_SIZE)
        );

        cl::Event sums_event = merge_sums(
                dev_input,
                dev_sum,
                dev_res
        );
        sums_event.wait();

        queue.enqueueReadBuffer(dev_res, CL_TRUE, 0, sizeof(float) * result.size(), result.data());
        return result;

    } catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }
}

int main() {
    std::vector<float> input;
    read_input(INPUT_FILE, input);
    size_t size = input.size();
    fill_zeros(input);
    std::vector<float> result = prefix_sum(input);
    write_result(OUTPUT_FILE, result, size);
    return 0;
}

