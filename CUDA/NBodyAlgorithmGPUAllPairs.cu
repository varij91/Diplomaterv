#include "NBodyAlgorithmGPUAllPairs.cuh"

void NBodyAlgorithmGPUAllPairs::unpackBodies(std::vector<Body> &bodies) {
#pragma unroll
    for (int i = 0; i < mp_properties->numBody; i++) {
 
    }
}


// Kellene egy k�l�n f�ggv�ny valahova magasabb szinten ami ellen�rzi, hogy van-e alkalmas GPU
// Kellene egy�b utility f�ggv�ny a device param�terek ki�rat�s�ra is, de ha lehet ne szemetelj�k tele a k�dot CUDA k�ddal
void NBodyAlgorithmGPUAllPairs::advance(std::vector<Body> &bodies) {

}

__global__ void advanceKernel(float3 *pos, float *mass, float3 *acc, float eps2) {

}