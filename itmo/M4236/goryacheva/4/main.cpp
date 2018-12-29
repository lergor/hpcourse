#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

const size_t GROUP_SIZE = 256;
const size_t BLOCK_SIZE = 256;
const int PRECISION = 3;
const int N_MAX = 1048576;

const std::string INPUT_FILE = "input.txt";
const std::string OUTPUT_FILE = "output.txt";


void read_input(std::string const &file, std::vector<float> &vec) {
    std::ifstream input(file.c_str());
    size_t N;
    input >> N;
    std::cout << N << '\n';

    vec = std::vector<float>(N);

    for (int i = 0; i < N; i++) {
        input >> vec[i];
        std::cout << vec[i] << " ";
    }
    std::cout << "\n";

    input.close();
}

void write_result(std::string const &file, std::vector<float> const &result) {
    std::ofstream output(file.c_str());
    for (float e : result) {
        output << e << " ";
        std::cout << e << " ";
    }
    std::cout << std::endl;
    output << std::endl;
    output.close();
}

size_t calculate_thread_size(size_t N) {
    size_t thread_size = BLOCK_SIZE;
    while (thread_size < N) {
        thread_size *= 2;
    }
    return thread_size;
}

size_t calculate_blocks_num(size_t thread_size) {
    size_t needed_memory = 1;
    while (needed_memory * BLOCK_SIZE < thread_size) {
        needed_memory++;
    }
    return needed_memory;
}

void extend_input(std::vector<float>& input, size_t size) {
    while (input.size() % size != 0) {
        input.push_back(0);
    }
}

int main()
{
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
         return 0;
      }

      std::vector<float> input;
      read_input(INPUT_FILE, input);
      std::vector<float> result(input.size(), 0);
      extend_input(input, GROUP_SIZE);
      std::vector<float> tmp(2 * input.size(), 0);


      // allocate device buffer to hold message
      cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
      cl::Buffer dev_res(context, CL_MEM_WRITE_ONLY, sizeof(float) * result.size());

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * input.size(), input.data());

      size_t thread_size = calculate_thread_size(input.size());
      std::cout << "thread size: " << thread_size << '\n';
      size_t blocks_num = calculate_blocks_num(thread_size);
      std::cout << "needed memory: " << blocks_num << '\n';

      queue.finish();

      cl::Kernel kernel_gmem(program, "gpu_prefix");
      cl::KernelFunctor prefix_gmem(
            kernel_gmem,
            queue,
            cl::NullRange,
            cl::NDRange(thread_size),
            cl::NDRange(BLOCK_SIZE)
      );

      cl::Event event = prefix_gmem(
                                  dev_input,
                                  dev_res,
                                  cl::__local(sizeof(float) * blocks_num * BLOCK_SIZE),
                                  cl::__local(sizeof(float) * blocks_num * BLOCK_SIZE),
                                  cl_int(input.size())
      );

      event.wait();

      queue.enqueueReadBuffer(dev_res, CL_TRUE, 0, sizeof(float) * result.size(), result.data());
      write_result(OUTPUT_FILE, result);
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}