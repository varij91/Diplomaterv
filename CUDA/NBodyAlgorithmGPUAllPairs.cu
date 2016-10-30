#include "NBodyAlgorithmGPUAllPairs.cuh"

void NBodyAlgorithmGPUAllPairs::unpackBodies(std::vector<Body> &bodies) {
#pragma unroll
    for (int i = 0; i < mp_properties->numBody; i++) {
 
    }
}


// Kellene egy külön függvény valahova magasabb szinten ami ellenõrzi, hogy van-e alkalmas GPU
// Kellene egyéb utility függvény a device paraméterek kiíratására is, de ha lehet ne szemeteljük tele a kódot CUDA kóddal
void NBodyAlgorithmGPUAllPairs::advance(std::vector<Body> &bodies) {

}

__global__ void advanceKernel(float3 *pos, float *mass, float3 *acc, float eps2) {

}