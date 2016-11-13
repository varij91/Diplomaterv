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
    if (!constMemoryInitalized) {
        checkCudaError(cudaMemcpyToSymbol(d_NUM_BODY, &(mp_properties->numBody), sizeof(int)));
        checkCudaError(cudaMemcpyToSymbol(d_POSITION_SCALE, &(mp_properties->positionScale), sizeof(int)));
        checkCudaError(cudaMemcpyToSymbol(d_EPS2, &(mp_properties->EPS2), sizeof(float)));
        checkCudaError(cudaMemcpyToSymbol(d_VELOCITY_DAMPENING, &(mp_properties->VELOCITY_DAMPENING), sizeof(float)));
        checkCudaError(cudaMemcpyToSymbol(d_STEP_TIME, &(mp_properties->stepTime), sizeof(float)));
        
        constMemoryInitalized = true;
    }

    if (mp_properties->mode == GUI) {
        //advanceKernelWithColor << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> > (mpd_position, mpd_mass, mpd_acceleration, mp_properties->numBody, mp_properties->EPS2, mpd_numNeighbours, mp_properties->positionScale);
        //integrateKernelWithColor << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> >(mpd_mass, mpd_position[1 - m_writeable], mpd_position[m_writeable], mpd_velocity, mpd_acceleration, mp_properties->numBody, mp_properties->EPS2, mp_properties->stepTime, mp_properties->VELOCITY_DAMPENING, mpd_numNeighbours, mp_properties->positionScale);

        integrateKernelWithFloat4WithColor << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> >(mpd_position4[1 - m_writeable], mpd_position4[m_writeable], mpd_velocity4, mpd_numNeighbours);
    }
    else {
        checkCudaError(cudaFuncSetCacheConfig(&integrateKernel, cudaFuncCachePreferShared));
        //integrateKernel << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> >(mpd_mass, mpd_position[1 - m_writeable], mpd_position[m_writeable], mpd_velocity, mpd_acceleration, mp_properties->numBody, mp_properties->EPS2, mp_properties->stepTime, mp_properties->VELOCITY_DAMPENING);
        //integrateKernelWithConst << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> >(mpd_mass, mpd_position[1 - m_writeable], mpd_position[m_writeable], mpd_velocity, mpd_acceleration);
        integrateKernelWithFloat4 << < m_gridSize, m_threadBlockSize, m_sharedMemorySize >> >(mpd_position4[1 - m_writeable], mpd_position4[m_writeable], mpd_velocity4);
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelStatus) << std::endl;
    }

    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaMemcpy(mph_position, mpd_position[m_writeable], mp_properties->numBody * sizeof(float3), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(mph_position4, mpd_position4[m_writeable], mp_properties->numBody * sizeof(float4), cudaMemcpyDeviceToHost));
    m_writeable = 1 - m_writeable; // érték invertálás, buffer váltása

    checkCudaError(cudaMemcpy(mph_velocity, mpd_velocity, mp_properties->numBody * sizeof(float3), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(mph_velocity4, mpd_velocity4, mp_properties->numBody * sizeof(float4), cudaMemcpyDeviceToHost));
    //checkCudaError(cudaMemcpy(mph_acceleration, mpd_acceleration, mp_properties->numBody * sizeof(float3), cudaMemcpyDeviceToHost));

    if (mp_properties->mode == GUI) {
        checkCudaError(cudaMemcpy(mph_numNeighbours, mpd_numNeighbours, mp_properties->numBody * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < mp_properties->numBody; i++) {
            mp_properties->numNeighbours.at(i) = (unsigned int)mph_numNeighbours[i];
        }
    }
    packBodies4(bodies);
    //updateBodies(bodies);
}

// Valamiért nem tetszik az NVCC-nek ha másik fájlban van deklarálva és definiálva a kernel által hívogatott __device__ függvény
__device__ float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ, float3 accSumI, const int eps2) {
    float3 r;
    
    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;
   /* if (!(blockIdx.x * blockDim.x + threadIdx.x))
        printf("gpu - r: (%f, %f, %f)\n", r.x, r.y, r.z);*/

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + eps2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;
    /*if (!(blockIdx.x * blockDim.x + threadIdx.x))
        printf("gpu - rabs, rabsinv, temp, massj (%f, %f, %f, %f)\n", rabs, rabsInv, temp, massJ);*/
    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktam a massJ-t, hogy ne kelljen mindig összeszorozni
    // Be kelle hozni az gyorulás értékek akkumulálását, mert float3/float4-gyel való mûveleteket nem támogatja a CUDA C
    // Szintén nem elhanyagolható hogy MAC mûveletet ki kell használni, nem ajánlott az akkumulálást külön elvégezni
    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;
    /*if (!(blockIdx.x * blockDim.x + threadIdx.x))
        printf("gpu - accsum: (%f, %f, %f)\n", accSumI.x, accSumI.y, accSumI.z);*/
    return accSumI;
}

__device__ float3 tileCalculateAcceleration(const float3 posI, float3 accI, const int eps2) {
    float3 accSumI = accI;

    /*extern __shared__ int buf[];
    when you launch the kernel you should launch it this way;
    kernel << <blocks, threads, numbytes_for_shared >> >(...);
    If you have multiple extern declaration of shared :
    extern __shared__ float As[];
    extern __shared__ float Bs[];
    this will lead to As pointing to the same address as Bs.
    You will need to keep As and Bs inside the 1D - array.*/

    /*extern __shared__ float sh_mass[];
    extern __shared__ float3 sh_pos[];*/
    extern __shared__ float sh_pos_mass[];
#pragma unroll 128
    for (int i = 0; i < blockDim.x; i++) {
        accSumI = calculateAcceleration(posI, sh_pos_mass[i], { sh_pos_mass[blockDim.x + (3 * i)], sh_pos_mass[blockDim.x + (3 * i) + 1], sh_pos_mass[blockDim.x + (3 * i) + 2] }, accSumI, eps2);
    }
    return accSumI;
}

// Virtuális mátrixként elképzelve a testeket
// Egy tile az indított thread block x paramétereitõl függ: általános esetben (256,1,1)
// Összes szál szám = testek száma
// Minden szál betölti a saját test paramétereit a regiszterekbe (ezzel a mátrix egyik oldala teljesen le van fedve)
// A shared memóriába töltögetésnél is, minden szál egy testet tölt be (ezzel a regiszterek és a shared memóriában rendelkezésre áll egy 256x256-os tile a számításra)

__device__ float3 advance(float3 posI, float *g_mass, float3 *g_pos, int g_numBodies, float g_eps2) {
    //extern __shared__ float sh_mass[];
    //extern __shared__ float3 sh_pos[];
    extern __shared__ float sh_pos_mass[];

    float3 accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < g_numBodies; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;

        sh_pos_mass[threadIdx.x] = g_mass[tileID];  // folytonos hozzáférés --> nincs bank konfliktus
        sh_pos_mass[blockDim.x + (3 * threadIdx.x)] = g_pos[tileID].x; // strided hozzáférés -> nincs bank konfliktus
        sh_pos_mass[blockDim.x + (3 * threadIdx.x) + 1] = g_pos[tileID].y;
        sh_pos_mass[blockDim.x + (3 * threadIdx.x) + 2] = g_pos[tileID].z;

        __syncthreads();    // shared memória töltése
        accI = tileCalculateAcceleration(posI, accI, g_eps2);
/*#pragma unroll 128
        for (int j = 0; j < blockDim.x; j++) {
            accI = calculateAcceleration(posI, sh_pos_mass[j], { sh_pos_mass[blockDim.x + (3 * j)], sh_pos_mass[blockDim.x + (3 * j) + 1], sh_pos_mass[blockDim.x + (3 * j) + 2] }, accI, g_eps2);
        }*/
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }

    return accI;
}

__device__ float3 advance_NoSharedNoTile(float3 posI, float *g_mass, float3 *g_pos, int g_numBodies, float g_eps2) {
    float3 accI = { 0.0f, 0.0f, 0.0f };

#pragma unroll 256
    for (int i = 0; i < g_numBodies; i++) {
        accI = calculateAcceleration(posI, g_mass[i], g_pos[i], accI, g_eps2);
    }
    return accI;
}

__global__ void integrateKernel(float *g_mass, float3 *g_posOld, float3 *g_posNew, float3 *g_vel, float3 *g_acc, int g_numBodies, float g_eps2, float g_stepTime, float g_velDampening) {

    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID > g_numBodies) return;

    float stepTime2 = 0.5f * g_stepTime * g_stepTime;

    float3 posI = g_posOld[globalThreadID]; // coalesced
    float3 velI = g_vel[globalThreadID];    // coalesced
    float3 accI;

    accI = advance(posI, g_mass, g_posOld, g_numBodies, g_eps2);
    //accI = advance_NoSharedNoTile(posI, g_mass, g_posOld, g_numBodies, g_eps2);

    /*float3 r;
    for (int i = 0; i < g_numBodies; i++) {
        r.x = g_posOld[i].x - posI.x;
        r.y = g_posOld[i].y - posI.y;
        r.z = g_posOld[i].z - posI.z;
 
        float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + g_eps2);
        float rabsInv = 1.0f / (rabs * rabs * rabs);
        float temp = g_mass[i] * rabsInv;

        accI.x += r.x * temp;
        accI.y += r.y * temp;
        accI.z += r.z * temp;
    }*/

    // Pozíció, sebesség és a szomszédos testek számának frissítése
    posI.x += velI.x * g_stepTime; + accI.x * stepTime2;
    posI.y += velI.y * g_stepTime; + accI.y * stepTime2;
    posI.z += velI.z * g_stepTime; + accI.z * stepTime2;

    velI.x = velI.x * g_velDampening + accI.x * g_stepTime;
    velI.y = velI.y * g_velDampening + accI.y * g_stepTime;
    velI.z = velI.z * g_velDampening + accI.z * g_stepTime;

    g_posNew[globalThreadID] = posI;
    g_vel[globalThreadID] = velI;
    //g_acc[globalThreadID] = accI;
}


__device__ float3 calculateAccelerationWithConst(const float3 posI, const float massJ, const float3 posJ, float3 accSumI) {
    float3 r;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + d_EPS2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;

    return accSumI;
}

__device__ float3 tileCalculateAccelerationWithConst(const float3 posI, float3 accI) {
    float3 accSumI = accI;

    extern __shared__ float sh_pos_mass[];
#pragma unroll 128
    for (int i = 0; i < blockDim.x; i++) {
        accSumI = calculateAccelerationWithConst(posI, sh_pos_mass[i], { sh_pos_mass[blockDim.x + (3 * i)], sh_pos_mass[blockDim.x + (3 * i) + 1], sh_pos_mass[blockDim.x + (3 * i) + 2] }, accSumI);
    }
    return accSumI;
}

__device__ float3 advanceWithConst(float3 posI, float *g_mass, float3 *g_pos) {
    extern __shared__ float sh_pos_mass[];

    float3 accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < d_NUM_BODY; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;

        sh_pos_mass[threadIdx.x] = g_mass[tileID];  // folytonos hozzáférés --> nincs bank konfliktus
        sh_pos_mass[blockDim.x + (3 * threadIdx.x)] = g_pos[tileID].x; // strided hozzáférés -> nincs bank konfliktus
        sh_pos_mass[blockDim.x + (3 * threadIdx.x) + 1] = g_pos[tileID].y;
        sh_pos_mass[blockDim.x + (3 * threadIdx.x) + 2] = g_pos[tileID].z;

        __syncthreads();    // shared memória töltése
        accI = tileCalculateAccelerationWithConst(posI, accI);
/*#pragma unroll 128
        for (int j = 0; j < blockDim.x; j++) {
        accI = calculateAcceleration(posI, sh_pos_mass[j], { sh_pos_mass[blockDim.x + (3 * j)], sh_pos_mass[blockDim.x + (3 * j) + 1], sh_pos_mass[blockDim.x + (3 * j) + 2] }, accI, g_eps2);
        }*/
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }

    return accI;
}

__global__ void integrateKernelWithConst(float *g_mass, float3 *g_posOld, float3 *g_posNew, float3 *g_vel, float3 *g_acc) {
    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID > d_NUM_BODY) return;

    float stepTime2 = 0.5f * d_STEP_TIME * d_STEP_TIME;

    float3 posI = g_posOld[globalThreadID]; // coalesced
    float3 velI = g_vel[globalThreadID];    // coalesced
    float3 accI;

    accI = advanceWithConst(posI, g_mass, g_posOld);
    //accI = advance_NoSharedNoTile(posI, g_mass, g_posOld, g_numBodies, g_eps2);

    /*float3 r;
    for (int i = 0; i < g_numBodies; i++) {
    r.x = g_posOld[i].x - posI.x;
    r.y = g_posOld[i].y - posI.y;
    r.z = g_posOld[i].z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + g_eps2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = g_mass[i] * rabsInv;

    accI.x += r.x * temp;
    accI.y += r.y * temp;
    accI.z += r.z * temp;
    }*/

    // Pozíció, sebesség és a szomszédos testek számának frissítése
    posI.x += velI.x * d_STEP_TIME; +accI.x * stepTime2;
    posI.y += velI.y * d_STEP_TIME; +accI.y * stepTime2;
    posI.z += velI.z * d_STEP_TIME; +accI.z * stepTime2;

    velI.x = velI.x * d_VELOCITY_DAMPENING + accI.x * d_STEP_TIME;
    velI.y = velI.y * d_VELOCITY_DAMPENING + accI.y * d_STEP_TIME;
    velI.z = velI.z * d_VELOCITY_DAMPENING + accI.z * d_STEP_TIME;

    g_posNew[globalThreadID] = posI;
    g_vel[globalThreadID] = velI;
    //g_acc[globalThreadID] = accI;
}


__device__ float3 calculateAccelerationWithFloat4(float4 posI, float4 posJ, float3 accSumI) {
    float3 r;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + d_EPS2);
    // sqrt kiszedésével közel 20 GFLOPS-os (50%-os) teljesítménybeli növekedés
    // SPU korlát lenne a gáz?
    //float rabs = r.x * r.x + r.y * r.y + r.z * r.z + d_EPS2;
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = posJ.w * rabsInv;

    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;

    return accSumI;
}

__device__ float3 advanceWithFloat4(float4 posI, float4 *g_pos) {
    extern __shared__ float4 sh_pm[];

    float3 accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < d_NUM_BODY; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;

        sh_pm[threadIdx.x] = g_pos[tileID];

        __syncthreads();    // shared memória töltése
#pragma unroll 128
        for (int j = 0; j < blockDim.x; j++) {
            accI = calculateAccelerationWithFloat4(posI, sh_pm[j], accI);
        }
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }

    return accI;
}

__device__ float3 advanceWithFloat4_NoSharedNoTile(float4 posI, float4 *g_pos) {
    float3 accI = { 0.0f, 0.0f, 0.0f };

#pragma unroll 128
    for (int i = 0; i < d_NUM_BODY; i++) {
        accI = calculateAccelerationWithFloat4(posI, g_pos[i], accI);
    }
    return accI;
}

__global__ void integrateKernelWithFloat4(float4 *g_posOld, float4 *g_posNew, float4 *g_vel) {
    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID > d_NUM_BODY) return;

    float stepTime2 = 0.5f * d_STEP_TIME * d_STEP_TIME;

    float4 posI = g_posOld[globalThreadID]; // coalesced
    float4 velI = g_vel[globalThreadID];    // coalesced
    float3 accI;

    accI = advanceWithFloat4(posI, g_posOld);
    //accI = advanceWithFloat4_NoSharedNoTile(posI, g_posOld);

    /*float3 r;
    for (int i = 0; i < g_numBodies; i++) {
    r.x = g_posOld[i].x - posI.x;
    r.y = g_posOld[i].y - posI.y;
    r.z = g_posOld[i].z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + g_eps2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = g_mass[i] * rabsInv;

    accI.x += r.x * temp;
    accI.y += r.y * temp;
    accI.z += r.z * temp;
    }*/

    // Pozíció, sebesség és a szomszédos testek számának frissítése
    posI.x += velI.x * d_STEP_TIME; +accI.x * stepTime2;
    posI.y += velI.y * d_STEP_TIME; +accI.y * stepTime2;
    posI.z += velI.z * d_STEP_TIME; +accI.z * stepTime2;

    velI.x = velI.x * d_VELOCITY_DAMPENING + accI.x * d_STEP_TIME;
    velI.y = velI.y * d_VELOCITY_DAMPENING + accI.y * d_STEP_TIME;
    velI.z = velI.z * d_VELOCITY_DAMPENING + accI.z * d_STEP_TIME;

    g_posNew[globalThreadID] = posI;
    g_vel[globalThreadID] = velI;
    //g_acc[globalThreadID] = accI;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
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
    extern __shared__ float sh_pos_mass[];
    for (int i = 0; i < blockDim.x; i++) {
        accSumI = calculateAccelerationWithColor(posI, sh_pos_mass[i], { sh_pos_mass[blockDim.x + (3 * i)], sh_pos_mass[blockDim.x + (3 * i) + 1], sh_pos_mass[blockDim.x + (3 * i) + 2] }, accSumI, eps2, numNeighbours, posScale);
    }
    return accSumI;
}

__device__ float3 advanceWithColor(float3 posI, float *g_mass, float3 *g_pos, int g_numBodies, float g_eps2, float *numNeighbours, float g_posScale) {
    extern __shared__ float sh_pos_mass[];

    float3 accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < g_numBodies; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;

        sh_pos_mass[threadIdx.x] = g_mass[tileID];
        sh_pos_mass[blockDim.x + (3 * threadIdx.x)] = g_pos[tileID].x;
        sh_pos_mass[blockDim.x + (3 * threadIdx.x) + 1] = g_pos[tileID].y;
        sh_pos_mass[blockDim.x + (3 * threadIdx.x) + 2] = g_pos[tileID].z;

        __syncthreads();    // shared memória töltése
        accI = tileCalculateAccelerationWithColor(posI, accI, g_eps2, numNeighbours, g_posScale);
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }

    return accI;
}

__global__ void integrateKernelWithColor(float *g_mass, float3 *g_posOld, float3 *g_posNew, float3 *g_vel, float3 *g_acc,
    int g_numBodies, float g_eps2, float g_stepTime, float g_velDampening, float *g_numNeighbours, float g_posScale) {
    
    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID >= g_numBodies) return;

    float stepTime2 = 0.5f * g_stepTime * g_stepTime;

    float3 posI = g_posOld[globalThreadID];
    float3 velI = g_vel[globalThreadID];
    float3 accI;

    float numNeighboursI = 0.0f;

    accI = advanceWithColor(posI, g_mass, g_posOld, g_numBodies, g_eps2, &numNeighboursI, g_posScale);

    // Pozíció, sebesség és a szomszédos testek számának frissítése
    posI.x += velI.x * g_stepTime + accI.x * stepTime2;
    posI.y += velI.y * g_stepTime + accI.y * stepTime2;
    posI.z += velI.z * g_stepTime + accI.z * stepTime2;

    velI.x = velI.x * g_velDampening + accI.x * g_stepTime;
    velI.y = velI.y * g_velDampening + accI.y * g_stepTime;
    velI.z = velI.z * g_velDampening + accI.z * g_stepTime;

    //printf("(%f, %f, %f)\n", accI.x, accI.y, accI.z);

    g_posNew[globalThreadID] = posI;
    g_vel[globalThreadID] = velI;
    // Kijelzéshez nem kell
    //g_acc[globalThreadID] = accI;
    g_numNeighbours[globalThreadID] = numNeighboursI;
}



__device__ float3 calculateAccelerationWithFloat4WithColor(float4 posI, float4 posJ, float3 accSumI, float *numNeighbours) {
    float3 r;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + d_EPS2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = posJ.w * rabsInv;
    (*numNeighbours) = (rabs < d_POSITION_SCALE) ? (*numNeighbours) + 1 : (*numNeighbours);

    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;

    return accSumI;
}

__device__ float3 advanceWithFloat4WithColor(float4 posI, float4 *g_pos, float *numNeighbours) {
    extern __shared__ float4 sh_pm[];

    float3 accI = { 0.0f, 0.0f, 0.0f };

    for (int i = 0, tile = 0; i < d_NUM_BODY; i += blockDim.x, tile++) {
        int tileID = tile * blockDim.x + threadIdx.x;

        sh_pm[threadIdx.x] = g_pos[tileID];

        __syncthreads();    // shared memória töltése
#pragma unroll 128
        for (int j = 0; j < blockDim.x; j++) {
            accI = calculateAccelerationWithFloat4WithColor(posI, sh_pm[j], accI, numNeighbours);
        }
        __syncthreads();    // ne kezdõdjön újra a shared memória feltöltése
    }

    return accI;
}

__global__ void integrateKernelWithFloat4WithColor(float4 *g_posOld, float4 *g_posNew, float4 *g_vel, float *g_numNeighbours) {
    int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadID > d_NUM_BODY) return;

    float stepTime2 = 0.5f * d_STEP_TIME * d_STEP_TIME;

    float4 posI = g_posOld[globalThreadID]; // coalesced
    float4 velI = g_vel[globalThreadID];    // coalesced
    float3 accI;

    float numNeighboursI = 0.0f;

    accI = advanceWithFloat4WithColor(posI, g_posOld, &numNeighboursI);

    // Pozíció, sebesség és a szomszédos testek számának frissítése
    posI.x += velI.x * d_STEP_TIME; +accI.x * stepTime2;
    posI.y += velI.y * d_STEP_TIME; +accI.y * stepTime2;
    posI.z += velI.z * d_STEP_TIME; +accI.z * stepTime2;

    velI.x = velI.x * d_VELOCITY_DAMPENING + accI.x * d_STEP_TIME;
    velI.y = velI.y * d_VELOCITY_DAMPENING + accI.y * d_STEP_TIME;
    velI.z = velI.z * d_VELOCITY_DAMPENING + accI.z * d_STEP_TIME;

    g_posNew[globalThreadID] = posI;
    g_vel[globalThreadID] = velI;
    //g_acc[globalThreadID] = accI;
    g_numNeighbours[globalThreadID] = numNeighboursI;
}