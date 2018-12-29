__kernel void gpu_prefix(
    __global float * input,
    __global float * result,
    __local float * r,
    __local float * s,
    int n
    ) {
    size_t block_size = get_local_size(0);
    size_t threads_size = get_global_size(0);
    size_t g_id = get_global_id(0);
    size_t l_id = get_local_id(0);

    int shift = 0;
    int extra_shift = 0;
    size_t step = 1;

    __local float* rrr;

    while(step < n)
    {
        r[shift + l_id] = input[extra_shift + g_id]; 
        s[shift + l_id] = input[extra_shift + g_id];
        barrier(CLK_LOCAL_MEM_FENCE);

        __local float* rr;
        __local float* ss;

        bool flag = true;
        rr = r;
        ss = s;

        for (size_t k = 1; k < block_size; k <<= 1)
        {
            if (step * g_id < n)
                ss[shift + l_id] = rr[shift + l_id] + (l_id > k - 1 ? rr[shift + l_id - k] : 0);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (flag) {
                rr = s; 
                ss = r;
            } else {
                rr = r; 
                ss = s;
            }
            flag = !flag;
        }

        extra_shift += threads_size / step;

        if ((l_id == 0) && (step * (g_id + block_size - 1) < n))
            input[extra_shift + g_id / block_size] = rr[shift + block_size - 1];
        barrier(CLK_GLOBAL_MEM_FENCE);

        shift += block_size;
        step *= block_size;

        rrr = rr;
    }

    while (step > 0)
    {
        shift -= block_size;
        if ((g_id >= block_size) && (step * g_id / block_size < n))
            rrr[shift + l_id] += input[extra_shift + g_id / block_size - 1];
        extra_shift -= threads_size / step;
        if (step * g_id < n)
            input[extra_shift + g_id] = rrr[shift + l_id];
        step /= block_size;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    result[g_id] = rrr[l_id];
}