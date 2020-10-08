__kernel void gpu_convolution(__global float * A, __global float * B, __global float * C, int N, int M)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	if (i >= N || j >= N)
		return;

	float value = 0;
	int HM = (M - 1) / 2;
	for (int k = -HM; k <= HM; k++) {
		if (i + k < 0 || i + k >= N) {
			continue;
		}
		for (int r = -HM; r <= HM; r++) {
			if (j + r < 0 || j + r >= N) {
				continue;
			}
			value += A[(i + k) * N + j + r] * B[(k + HM) * M + r + HM];
		}
	}
	C[i * N + j] = value;
}