__kernel void gpu_prefix(
    __global float * input, 
    __global float * output, 
    __local float * fst_buf, 
    __local float * snd_buf
) {
    uint g_id = get_global_id(0);
    uint l_id = get_local_id(0);
    uint block_size = get_local_size(0);

    fst_buf[l_id] = snd_buf[l_id] = input[g_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    __local float* ff;
    __local float* ss;

    bool flag = true;
    ff = fst_buf;
    ss = snd_buf;
 
    for(uint s = 1; s < block_size; s <<= 1) {
        ss[l_id] = ff[l_id] + (l_id > (s - 1) ? ff[l_id - s] : 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (flag) {
            ff = snd_buf;
            ss = fst_buf;
        } else {
            ff = fst_buf;
            ss = snd_buf;
        }

        flag = !flag;
    }
    output[g_id] = ff[l_id];
}

__kernel void gpu_merge(
    __global float* input,
    __global float* sums,
    __global float* output
) {
    uint l_id = get_local_id(0);
    uint gr_id = get_group_id(0);

    uint block_size = get_local_size(0);

    if (gr_id > 0) {
        uint i = l_id + gr_id * block_size;
        output[i] = input[i] + sums[gr_id - 1];
    }
} 