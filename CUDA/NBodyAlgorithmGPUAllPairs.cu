#include "NBodyAlgorithmGPUAllPairs.cuh"

void NBodyAlgorithmGPUAllPairs::updateBodies(std::vector<Body> &bodies) {
    float stepTime2 = mp_properties->stepTime * mp_properties->stepTime;
#pragma unroll
    for (int i = 0; i < mp_properties->numBody; i++) {
        mph_position[i].x += mph_velocity[i].x * mp_properties->stepTime + mph_acceleration[i].x * stepTime2;
        mph_position[i].y += mph_velocity[i].y * mp_properties->stepTime + mph_acceleration[i].y * stepTime2;
        mph_position[i].z += mph_velocity[i].z * mp_properties->stepTime + mph_acceleration[i].z * stepTime2;

        mph_velocity[i].x = mph_velocity[i].x * mp_properties->VELOCITY_DAMPENING + mph_acceleration[i].x * mp_properties->stepTime;
        mph_velocity[i].y = mph_velocity[i].y * mp_properties->VELOCITY_DAMPENING + mph_acceleration[i].y * mp_properties->stepTime;
        mph_velocity[i].z = mph_velocity[i].z * mp_properties->VELOCITY_DAMPENING + mph_acceleration[i].z * mp_properties->stepTime;
    }
    if (mp_properties->mode == GUI) {
        packBodies(bodies);
        for (int i = 0; i < mp_properties->numBody; i++) {
            mp_properties->numNeighbours.at(i) = (unsigned int)mph_numNeighbours[i];
        }
    }
}

// Kellene egy külön függvény valahova magasabb szinten ami ellenõrzi, hogy van-e alkalmas GPU
// Kellene egyéb utility függvény a device paraméterek kiíratására is, de ha lehet ne szemeteljük tele a kódot CUDA kóddal
void NBodyAlgorithmGPUAllPairs::advance(std::vector<Body> &bodies) {
    if (mp_properties->mode == GUI) {
        //advanceKernelWithColor << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> > (mpd_position, mpd_mass, mpd_acceleration, mp_properties->numBody, mp_properties->EPS2, mpd_numNeighbours, mp_properties->positionScale);
        integrateKernelWithColor << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> >(mpd_mass, mpd_position, mpd_velocity, mpd_acceleration, mp_properties->numBody, mp_properties->EPS2, mp_properties->stepTime, mp_properties->VELOCITY_DAMPENING, mpd_numNeighbours, mp_properties->positionScale);
    }
    else {
        //advanceKernel << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> > (mpd_position, mpd_mass, mpd_acceleration, mp_properties->numBody, mp_properties->EPS2);
        integrateKernel << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> >(mpd_mass, mpd_position, mpd_velocity, mpd_acceleration, mp_properties->numBody, mp_properties->EPS2, mp_properties->stepTime, mp_properties->VELOCITY_DAMPENING);
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelStatus) << std::endl;
    }

    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaMemcpy(mph_position, mpd_position, mp_properties->numBody * sizeof(float3), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(mph_velocity, mpd_velocity, mp_properties->numBody * sizeof(float3), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(mph_acceleration, mpd_acceleration, mp_properties->numBody * sizeof(float3), cudaMemcpyDeviceToHost));
    if (mp_properties->mode == GUI) {
        checkCudaError(cudaMemcpy(mph_numNeighbours, mpd_numNeighbours, mp_properties->numBody * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < mp_properties->numBody; i++) {
            mp_properties->numNeighbours.at(i) = (unsigned int)mph_numNeighbours[i];
        }
    }
    packBodies(bodies);
    //updateBodies(bodies);
}

// Valamiért nem tetszik az NVCC-nek ha másik fájlban van deklarálva és definiálva a kernel által hívogatott __device__ függvény
__device__ float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ, float3 accSumI, const int eps2) {
    float3 r;
    
    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + eps2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktam a massJ-t, hogy ne kelljen mindig összeszorozni
    // Be kelle hozni az gyorulás értékek akkumulálását, mert float3/float4-gyel való mûveleteket nem támogatja a CUDA C
    // Szintén nem elhanyagolható hogy MAC mûveletet ki kell használni, nem ajánlott az akkumulálást külön elvégezni
    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;
    return accSumI;
}

__device__ float3 tileCalculateAcceleration(const float3 posI, float3 accI, const int eps2) {
    float3 accSumI = accI;
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];
    for (int i = 0; i < blockDim.y; i++) {
        accSumI = calculateAcceleration(posI, sh_mass[i], sh_pos[i], accSumI, eps2);
    }
    return accSumI;
}

// Virtuális mátrixként elképzelve a testeket
// Egy tile az indított thread block x paramétereitõl függ: általános esetben (256,1,1)
// Összes szál szám = testek száma
// Minden szál betölti a saját test paramétereit a regiszterekbe (ezzel a mátrix egyik oldala teljesen le van fedve)
// A shared memóriába töltögetésnél is, minden szál egy testet tölt be (ezzel a regiszterek és a shared memóriában rendelkezésre áll egy 256x256-os tile a számításra)
__global__ void advanceKernel(float3 *g_pos, float *g_mass, float3 *g_acc, int g_numBodies, float g_eps2) {
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];

    //float massI;
    float3 posI;
    float3 accI;
    int numBodies = g_numBodies;
    int eps2 = g_eps2;

    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID > g_numBodies) return;
    //massI = g_mass[gridID];
    posI = g_pos[globalThreadID];
    accI = { 0.0f, 0.0f, 0.0f };
    //printf("%d\n", blockDim.y);
    for (int i = 0, tile = 0; i < numBodies; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;
        //if (threadIdx.x == 0 && threadIdx.y == 0) printf("%d, %d\n", i, tile);
        sh_mass[threadIdx.x] = g_mass[tileID];
        sh_pos[threadIdx.x] = g_pos[tileID];

        __syncthreads();    // shared memória töltése
        accI = tileCalculateAcceleration(posI, accI, eps2);
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }
    //printf("(%f, %f, %f)", accI.x, accI.y, accI.z);
    g_acc[globalThreadID] = accI;
}

__device__ float3 calculateAccelerationWithColor(const float3 posI, const float massJ, const float3 posJ, float3 accSumI, const int eps2, float *numNeighbours, const float posScale) {
    float3 r;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + eps2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;
    (*numNeighbours) = (rabs < posScale) ? (*numNeighbours) + 1 : (*numNeighbours);

    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktam a massJ-t, hogy ne kelljen mindig összeszorozni
    // Be kelle hozni az gyorulás értékek akkumulálását, mert float3/float4-gyel való mûveleteket nem támogatja a CUDA C
    // Szintén nem elhanyagolható hogy MAC mûveletet ki kell használni, nem ajánlott az akkumulálást külön elvégezni
    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;
    return accSumI;
}




__device__ float3 tileCalculateAccelerationWithColor(const float3 posI, float3 accI, const int eps2, float *numNeighbours, const float posScale) {
    float3 accSumI = accI;
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];
    for (int i = 0; i < blockDim.x; i++) {
        accSumI = calculateAccelerationWithColor(posI, sh_mass[i], sh_pos[i], accSumI, eps2, numNeighbours, posScale);
    }
    return accSumI;
}

__global__ void advanceKernelWithColor(float3 *g_pos, float *g_mass, float3 *g_acc, int g_numBodies, float g_eps2, float *g_numNeighbours, float g_posScale) {
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];

    //float massI;
    float3 posI;
    float3 accI;
    int numBodies = g_numBodies;
    int eps2 = g_eps2;
    int posScale = g_posScale;
    float numNeighbours = 0.0f;

    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID > g_numBodies) return;

    posI = g_pos[globalThreadID];
    accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < numBodies; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;

        sh_mass[threadIdx.x] = g_mass[tileID];
        sh_pos[threadIdx.x] = g_pos[tileID];

        __syncthreads();    // shared memória töltése
        accI = tileCalculateAccelerationWithColor(posI, accI, eps2, &numNeighbours, posScale);
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }
    g_acc[globalThreadID] = accI;
    g_numNeighbours[globalThreadID] = numNeighbours;
}




__device__ float3 advance(float3 posI, float *g_mass, float3 *g_pos, int g_numBodies, float g_eps2) {
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];

    float3 accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < g_numBodies; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;

        sh_mass[threadIdx.x] = g_mass[tileID];
        sh_pos[threadIdx.x] = g_pos[tileID];

        __syncthreads();    // shared memória töltése
        accI = tileCalculateAcceleration(posI, accI, g_eps2);
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }

    return accI;
}

__global__ void integrateKernel(float *g_mass, float3 *g_pos, float3 *g_vel, float3 *g_acc,
    int g_numBodies, float g_eps2, float g_stepTime, float g_velDampening) {

    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID > g_numBodies) return;

    float stepTime2 = 0.5f * g_stepTime * g_stepTime;

    float3 posI = g_pos[globalThreadID];
    float3 velI = g_vel[globalThreadID];
    float3 accI;

    float numNeighboursI = 0.0f;

    //int numBodies = g_numBodies;
    //int eps2 = g_eps2;
    //int posScale = g_posScale;

    accI = advance(posI, g_mass, g_pos, g_numBodies, g_eps2);

    // Pozíció, sebesség és a szomszédos testek számának frissítése
    posI.x += velI.x * g_stepTime + accI.x * stepTime2;
    posI.y += velI.y * g_stepTime + accI.y * stepTime2;
    posI.z += velI.z * g_stepTime + accI.z * stepTime2;

    velI.x = velI.x * g_velDampening + accI.x * g_stepTime;
    velI.y = velI.y * g_velDampening + accI.y * g_stepTime;
    velI.z = velI.z * g_velDampening + accI.z * g_stepTime;

    g_pos[globalThreadID] = posI;
    g_vel[globalThreadID] = velI;
    g_acc[globalThreadID] = accI;
}










__device__ float3 advanceWithColor(float3 posI, float *g_mass, float3 *g_pos, int g_numBodies, float g_eps2, float *numNeighbours, float g_posScale) {
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];

    float3 accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < g_numBodies; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;

        sh_mass[threadIdx.x] = g_mass[tileID];
        sh_pos[threadIdx.x] = g_pos[tileID];

        __syncthreads();    // shared memória töltése
        accI = tileCalculateAccelerationWithColor(posI, accI, g_eps2, numNeighbours, g_posScale);
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }

    return accI;
}

__global__ void integrateKernelWithColor(float *g_mass, float3 *g_pos, float3 *g_vel, float3 *g_acc,
    int g_numBodies, float g_eps2, float g_stepTime, float g_velDampening, float *g_numNeighbours, float g_posScale) {
    
    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID > g_numBodies) return;

    float stepTime2 = 0.5f * g_stepTime * g_stepTime;

    float3 posI = g_pos[globalThreadID];
    float3 velI = g_vel[globalThreadID];
    float3 accI;

    float numNeighboursI = 0.0f;

    //int numBodies = g_numBodies;
    //int eps2 = g_eps2;
    //int posScale = g_posScale;

    accI = advanceWithColor(posI, g_mass, g_pos, g_numBodies, g_eps2, &numNeighboursI, g_posScale);

    // Pozíció, sebesség és a szomszédos testek számának frissítése
    posI.x += velI.x * g_stepTime + accI.x * stepTime2;
    posI.y += velI.y * g_stepTime + accI.y * stepTime2;
    posI.z += velI.z * g_stepTime + accI.z * stepTime2;

    velI.x = velI.x * g_velDampening + accI.x * g_stepTime;
    velI.y = velI.y * g_velDampening + accI.y * g_stepTime;
    velI.z = velI.z * g_velDampening + accI.z * g_stepTime;

    //printf("(%f, %f, %f)\n", accI.x, accI.y, accI.z);

    g_pos[globalThreadID] = posI;
    g_vel[globalThreadID] = velI;
    g_acc[globalThreadID] = accI;
    g_numNeighbours[globalThreadID] = numNeighboursI;
}

