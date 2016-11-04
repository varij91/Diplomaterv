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
        std::vector<unsigned int> temp(mph_numNeighbours, mph_numNeighbours + mp_properties->numBody);
        mp_properties->numNeighbours = temp;
    }
}

// Kellene egy külön függvény valahova magasabb szinten ami ellenõrzi, hogy van-e alkalmas GPU
// Kellene egyéb utility függvény a device paraméterek kiíratására is, de ha lehet ne szemeteljük tele a kódot CUDA kóddal
void NBodyAlgorithmGPUAllPairs::advance(std::vector<Body> &bodies) {
    if (mp_properties->mode == GUI) {
        //advanceKernelWithColor << < m_gridSize, m_threadBlockSize, m_sharedMemorySize + 12 >> > (mpd_position, mpd_mass, mpd_acceleration, mpd_numNeighbours);
    }
    else {
        advanceKernel << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> > (mpd_position, mpd_mass, mpd_acceleration, mpd_numBodies, mpd_eps2);
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelStatus) << std::endl;
    }

    //ITT FAGY KI
    checkCudaError(cudaDeviceSynchronize());

    // TODO: visszaolvasni a kernel által kiszámolt gyorsulásértékeket
    // Update-elni a sebességet és a pozíciót

    checkCudaError(cudaMemcpy(mph_acceleration, mpd_acceleration, mp_properties->numBody * sizeof(float3), cudaMemcpyDeviceToHost));
    if (mp_properties->mode == GUI) {
        checkCudaError(cudaMemcpy(mph_numNeighbours, mpd_numNeighbours, mp_properties->numBody * sizeof(float), cudaMemcpyDeviceToHost));
    }
    updateBodies(bodies);
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
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
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
    for (int i = 0; i < blockDim.x; i++) {
        accSumI = calculateAcceleration(posI, sh_mass[i], sh_pos[i], accSumI, eps2);
    }
    return accSumI;
}

__global__ void advanceKernel(float3 *g_pos, float *g_mass, float3 *g_acc, int *g_numBodies, float *g_eps2) {
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];

    //float massI;
    float3 posI;
    float3 accI;
    int numBodies = (*g_numBodies);
    int eps2 = (*g_eps2);

    int gridID = blockIdx.x * blockDim.x + threadIdx.x;
    //massI = g_mass[gridID];
    posI = g_pos[gridID];
    accI = { 0.0f, 0.0f, 0.0f };
    printf("111111111\n");
    for (int i = 0, tile = 0; i < numBodies; i += blockDim.y, tile++) {
        int tileID = tile * blockDim.y + threadIdx.x;
        printf("222222222\n");
        // Ha esetleg nem négyzetes TB lenne, nem kell minden threadnek betöltenie a shared memória
        if (threadIdx.x < blockDim.y) {
            sh_mass[threadIdx.x] = g_mass[tileID];
            sh_pos[threadIdx.x] = g_pos[tileID];
            //printf("(%f, %f, %f)", sh_pos[threadIdx.x].x, sh_pos[threadIdx.x].y, sh_pos[threadIdx.x].z);
           
        }
        __syncthreads();    // shared memória töltése
        accI = tileCalculateAcceleration(posI, accI, eps2);
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }
    //printf("(%f, %f, %f)", accI.x, accI.y, accI.z);
    g_acc[gridID] = accI;
}


/*__device__ float3 calculateAccelerationWithColor(const float3 posI, const float massJ, const float3 posJ, float3 accSumI, float *numNeighbours) {
    float3 r;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + d_EPS2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;
    (*numNeighbours) = (rabs < d_POSITION_SCALE) ? (*numNeighbours) + 1 : (*numNeighbours);
    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
    // Be kelle hozni az gyorulás értékek akkumulálását, mert float3/float4-gyel való mûveleteket nem támogatja a CUDA C
    // Szintén nem elhanyagolható hogy MAC mûveletet ki kell használni, nem ajánlott az akkumulálást külön elvégezni
    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;
    return accSumI;
}

__device__ float3 tileCalculateAccelerationWithColor(const float3 posI, float3 accI, float *numNeighbours) {
    float3 accSumI = accI;
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];
    for (int i = 0; i < blockDim.x; i++) {
        accSumI = calculateAccelerationWithColor(posI, sh_mass[i], sh_pos[i], accSumI, numNeighbours);
    }
    return accSumI;
}

__global__ void advanceKernelWithColor(float3 *g_pos, float *g_mass, float3 *g_acc, float *g_numNeighbours) {
    extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];

    //float massI;
    float3 posI;
    float3 accI;
    float numNeighbours = 0.0f;

    int gridID = blockIdx.x * blockDim.x + threadIdx.x;
    //massI = g_mass[gridID];
    posI = g_pos[gridID];
    accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < d_NUM_BODY; i += blockDim.y, tile++) {
        int tileID = tile * blockDim.y + threadIdx.x;
        // Ha esetleg nem négyzetes TB lenne, nem kell minden threadnek betöltenie a shared memória
        if (threadIdx.x < blockDim.y) {
            sh_mass[threadIdx.x] = g_mass[tileID];
            sh_pos[threadIdx.x] = g_pos[tileID];
        }
        __syncthreads();    // shared memória töltése
        accI = tileCalculateAccelerationWithColor(posI, accI, &numNeighbours);
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }
    g_acc[gridID] = accI;
}*/
