// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program 1.3 gpusum
// 
// RTX 2070
// C:\Users\Richard\OneDrive\toGit2>bin\gpusum.exe 1000000000 1000
// gpu sum = 2.0000000134, steps 1000000000 terms 1000 time 1881.113 ms
// 
// RTX 3080
// C:\Users\Richard\OneDrive\toGit2>bin\gpusum.exe 1000000000 1000
// gpu sum = 1.9999998123, steps 1000000000 terms 1000 time 726.253 ms

#include "cx.h"
#include <curand.h>
#include "cxtimers.h"              // cx timers
#include "cudamacro.h"

__host__ __device__ inline float sinsum(float x,int terms)
{
	float x2 = x*x;
	float term = x;   // first term of series
	float sum = term; // sum of terms so far
	for(int n = 1; n < terms; n++){
		term *= -x2 / (2*n*(2*n+1));  // build factorial
		sum += term;
	}
	return sum;
}

__global__ void gpu_sin(float *sums,int steps,int terms,float step_size)
{
	int step = blockIdx.x*blockDim.x+threadIdx.x; // unique thread ID
	if(step<steps){
		float x = step_size*step;
		sums[step] = sinsum(x,terms);  // store sin values in array
	}
}

__host__ __device__ inline float copyEnergy(float x)
{
	return x;
}

__global__ void initialize_spin_energy(float* spin_energy, Color color, 
                               const signed char* __restrict__ lattice,
                               const long long nx,
                               const long long ny) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int i = tid / ny;
  const int j = tid % ny;

  if (i >= nx || j >= ny) return;

  // Set stencil indices with periodicity
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int ip2 = (i + 2 < nx) ? i + 2 : i + 2 - nx;
  int inn = (i - 1 >= 0) ? i - 1: nx - 1;
  int in2 = (i - 2 >= 0) ? i - 2 : i - 2 + ny;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1: ny - 1;



  // Compute sum of nearest neighbor spins

  signed char nn_sum;
  nn_sum = J1*(lattice[inn * ny + j] + lattice[ipp * ny + j]) +  // vizinho 1 vertical
                      J2*(lattice[ip2 * ny + j] + lattice[in2 * ny + j]) +  // vizinho 2 vertical
                      J0*(lattice[i * ny + jpp] + lattice[i * ny + jnn]);   // vizinho 1 horizontal

  spin_energy[(i*ny + j)] = copyEnergy(nn_sum);
}

// Initialize lattice spins
__global__ void init_spins(signed char* lattice,
                           const float* __restrict__ randvals,
                           const long long nx,
                           const long long ny) {
  const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= nx * ny) return;

  float randval = randvals[tid];
  signed char val = (randval < 0.5f) ? -1 : 1;
  lattice[tid] = val;
}



int main(int argc,char *argv[])
{
	// float alpha = atof(argv[1]);
	// float t = atof(argv[2]);
	// float t_end = atof(argv[3]);
	// float step = atof(argv[4]);
	// char* fileName = argv[5];
	// long long ny = atoll(argv[6]);
	// int niters = atoi(argv[7]);
	float alpha = 0.376f;
	float t = 0.6f;
	char* fileName = "0.376_fim.txt";
	long long ny = 240;
	int niters = 100000;
	// Defaults
	long long nx = 10;
	//long long ny = 12;
	//float alpha = 0.1f;
	int nwarmup = 100;
	bool write = false;
	unsigned long long seed = 1234ULL;

	int steps = (argc > 1) ? atoi(argv[1]) : 10000000; // get command
	int terms = (argc > 2) ? atoi(argv[2]) : 1000;     // line arguments
	int threads = 256;
	int blocks = (steps+threads-1)/threads;  // ensure threads*blocks ≥ steps

	double pi = 3.14159265358979323;
	double step_size = pi / (nx*ny-1); // NB n-1 steps between n points
	
	curandGenerator_t rng;
	CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
	
	float *randvals;
  	CHECK_CUDA(cudaMalloc(&randvals, (nx * ny) * sizeof(*randvals)));

	signed char *lattice;
  	CHECK_CUDA(cudaMalloc(&lattice, (nx * ny) * sizeof(*lattice)));

	int blocks = (nx * ny/ + THREADS - 1) / THREADS;
	CHECK_CURAND(curandGenerateUniform(rng, randvals, (nx*ny)));
	init_spins<<<blocks, THREADS>>>(lattice, randvals, nx, ny);

	
	

	thrust::device_vector<float> dsums(nx*ny);         // GPU buffer 
	float *dptr = thrust::raw_pointer_cast(&dsums[0]); // get pointer

	initialize_spin_energy<<<blocks, THREADS>>>(dptr, Color::WHITE, lattice, nx, ny);
	cx::timer tim;
	//gpu_sin<<<blocks,threads>>>(dptr,nx*ny,terms,(float)step_size);
	double gpu_sum[2];
	gpu_sum[0] = thrust::reduce(dsums.begin(),dsums.end());
	double gpu_time = tim.lap_ms(); // get elapsed time

	// Trapezoidal Rule Correction
	gpu_sum[0] -= 0.5*(sinsum(0.0f,terms)+sinsum(pi,terms));
	gpu_sum[0] *= step_size;
	printf("gpu sum = %.10f, steps %d terms %d time %.3f ms\n",
		gpu_sum[0],steps,terms,gpu_time);
	return 0;
}
