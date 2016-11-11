#include "NBodyAlgorithmGPU.cuh"

// szopáshegy van az eps2 átadásával (nem akarom argumentumként átadni minden egyes alkalommal


void NBodyAlgorithmGPU::unpackBodies(std::vector<Body> &bodies) {
#pragma unroll
    for (int i = 0; i < mp_properties->numBody; i++) {
        mph_mass[i] = bodies.at(i).mass;
        mph_position[i] = bodies.at(i).position;
        mph_velocity[i] = bodies.at(i).velocity;
        mph_acceleration[i] = bodies.at(i).acceleration;
    }
}

void NBodyAlgorithmGPU::unpackBodies4(std::vector<Body> &bodies) {
#pragma unroll
    for (int i = 0; i < mp_properties->numBody; i++) { 
        mph_position4[i].x = bodies.at(i).position.x;
        mph_position4[i].y = bodies.at(i).position.y;
        mph_position4[i].z = bodies.at(i).position.z;
        mph_position4[i].w = bodies.at(i).mass;

        mph_velocity4[i].x = bodies.at(i).velocity.x;
        mph_velocity4[i].y = bodies.at(i).velocity.y;
        mph_velocity4[i].z = bodies.at(i).velocity.z;
        mph_velocity4[i].w = 0.0f;
    }
}

void NBodyAlgorithmGPU::packBodies(std::vector<Body> &bodies) {
#pragma unroll
    for (int i = 0; i < mp_properties->numBody; i++) {
        //bodies.at(i).mass = mph_mass[i];
        bodies.at(i).position = mph_position[i];
        bodies.at(i).velocity = mph_velocity[i];
        bodies.at(i).acceleration = mph_acceleration[i];
    }
}

void NBodyAlgorithmGPU::init(std::vector<Body> &bodies) {
    // Van-e CUDA kompatibilis GPU?
    int numDevice;
    checkCudaError(cudaGetDeviceCount(&numDevice));
    if (!numDevice) {
        std::cerr << "No CUDA compatible device detected. Aborting..." << std::endl;
        exit(0);
    }

    for (int i = 0; i < numDevice; i++) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, i));
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "   Device name: " << prop.name << std::endl;
        std::cout << "   Device Clock Rate (MHz): " << prop.clockRate / 1e3 << std::endl;
        std::cout << "   Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << std::endl;
        std::cout << "   Total Global Memory (MB): " << prop.totalGlobalMem / 1024 / 1024 << std::endl;
        std::cout << "   Total Constant Memory (kB): " << prop.totalConstMem / 1024 << std::endl;
        std::cout << "   Memory Clock Rate (MHz): " << prop.memoryClockRate / 1e3 << std::endl;
        std::cout << "   Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "   Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
        std::cout << std::endl;
        std::cout << "   Max Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << " )" << std::endl;
        std::cout << "   Max Threads Dimension: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << " )" << std::endl;
        std::cout << "   Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "   Max Threads Per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << std::endl;
        std::cout << "   Registers Per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "   Shared Memory per Block: " << prop.sharedMemPerBlock << std::endl;
    }
    
    // GPU kofiguráció
    checkCudaError(cudaSetDevice(0));
    //checkCudaError(cudaFuncSetCacheConfig(&matMulNaive, cudaFuncCachePreferL1));
    //checkCudaError(cudaFuncSetCacheConfig(&matMulNaive, cudaFuncCachePreferShared));

    // Testek kicsomagolása tömbökbe, a host memóriába
    unpackBodies(bodies);
    unpackBodies4(bodies);

    // Memóriaallokáció
    checkCudaError(cudaMalloc((void**)&mpd_mass, mp_properties->numBody * sizeof(float)));
    checkCudaError(cudaMalloc((void**)&mpd_position[0], mp_properties->numBody * sizeof(float3)));
    checkCudaError(cudaMalloc((void**)&mpd_position[1], mp_properties->numBody * sizeof(float3)));
    checkCudaError(cudaMalloc((void**)&mpd_velocity, mp_properties->numBody * sizeof(float3)));
    checkCudaError(cudaMalloc((void**)&mpd_acceleration, mp_properties->numBody * sizeof(float3)));

    checkCudaError(cudaMalloc((void**)&mpd_position4[0], mp_properties->numBody * sizeof(float4)));
    checkCudaError(cudaMalloc((void**)&mpd_position4[1], mp_properties->numBody * sizeof(float4)));
    checkCudaError(cudaMalloc((void**)&mpd_velocity4, mp_properties->numBody * sizeof(float4)));

    if (mp_properties->mode == GUI) {
        checkCudaError(cudaMalloc((void**)&mpd_numNeighbours, mp_properties->numBody * sizeof(float)));
    }
    
    // Másolás GPU global memóriába
    checkCudaError(cudaMemcpy(mpd_mass, mph_mass, mp_properties->numBody * sizeof(float), cudaMemcpyHostToDevice));
    // A positionNew inicializálatlan terület. A kernel fog neki mindig értéket adni
    checkCudaError(cudaMemcpy(mpd_position[1 - m_writeable], mph_position, mp_properties->numBody * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(mpd_velocity, mph_velocity, mp_properties->numBody * sizeof(float3), cudaMemcpyHostToDevice));

    checkCudaError(cudaMemcpy(mpd_position4[1 - m_writeable], mph_position4, mp_properties->numBody * sizeof(float4), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(mpd_velocity4, mph_velocity4, mp_properties->numBody * sizeof(float4), cudaMemcpyHostToDevice));

    //checkCudaError(cudaMemcpyToSymbol(d_NUM_BODY, &(mp_properties->numBody), sizeof(mp_properties->numBody)));
    //checkCudaError(cudaMemcpyToSymbol(d_POSITION_SCALE, (float*)(&(mp_properties->positionScale)), sizeof(float)));
    //checkCudaError(cudaMemcpyToSymbol(d_EPS2, &(mp_properties->EPS2), sizeof(mp_properties->EPS2)));

    setKernelParameters();
}

void NBodyAlgorithmGPU::destroy() {
    // Allokált memória felszabadítása
    checkCudaError(cudaFree(mpd_mass));
    checkCudaError(cudaFree(mpd_position[0]));
    checkCudaError(cudaFree(mpd_position[1]));
    checkCudaError(cudaFree(mpd_velocity));
    checkCudaError(cudaFree(mpd_acceleration));

    checkCudaError(cudaFree(mpd_position4[0]));
    checkCudaError(cudaFree(mpd_position4[1]));
    checkCudaError(cudaFree(mpd_velocity4));

    if (mp_properties->mode == GUI) {
        checkCudaError(cudaFree(mpd_numNeighbours));
    }
    checkCudaError(cudaDeviceReset());
}

void NBodyAlgorithmGPU::setKernelParameters() {
    int numBody = mp_properties->numBody;
    float minOccupancy = 75.0f; // %

    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0));

    int numMultiProcessor = prop.multiProcessorCount;
    int maxResidentThreadBlockPerMultiProcessor;
    switch (prop.major) // Compute capability
    {
    case 2: // 2.x
        maxResidentThreadBlockPerMultiProcessor = 8;
        break;
    case 3:
        maxResidentThreadBlockPerMultiProcessor = 16;
        break;
    case 5:
        maxResidentThreadBlockPerMultiProcessor = 32;
        break;
    case 6:
        maxResidentThreadBlockPerMultiProcessor = 32;
        break;
    default:
        maxResidentThreadBlockPerMultiProcessor = 8;
        break;
    }
    
    int maxActiveThreadBlocks = maxResidentThreadBlockPerMultiProcessor * numMultiProcessor;    // 16
    int maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    int maxActiveThreads = maxThreadsPerMultiProcessor * numMultiProcessor;     // 3072
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;       // 1024
    int wrapSize = prop.warpSize;   // 32

    uint3 tempDim = { prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] };
    dim3 maxGridDim = tempDim;
    tempDim = { prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] };
    dim3 maxThreadDim = tempDim;

    int maxSharedMemoryPerBlock = prop.sharedMemPerBlock;   // összesen 48k
    int maxRegistersPerBlock = prop.regsPerBlock;           // összesen 32768
    
    // Compute Capability:      2.1
    // Max register per block:  32768
    // Max register per thread: 63
    
    int optimalThreadsPerBlock = maxActiveThreads / minOccupancy * 100 / maxActiveThreadBlocks; // 256
    //optimalThreadsPerBlock = 128;
    // 16x16-os kernel indításának nincs nagyon értelme ezzel a tile-os, kommunikáció nélküli módszerrel
    // Egydimenziós kiosztással az Y és Z értékét fixen 1-re állítom
    unsigned int threadBlockX = optimalThreadsPerBlock;    // 256
    unsigned int threadBlockY = 1;
    unsigned int threadBlockZ = 1;

    unsigned int blockGridX = numBody / threadBlockX + ((numBody % threadBlockX) != 0);
    unsigned int blockGridY = 1;
    unsigned int blockGridZ = 1;
    if (threadBlockX > maxThreadDim.x || threadBlockY > maxThreadDim.y || threadBlockZ > maxThreadDim.z) {
        std::cout << "Thread blocks contain more threads than the max value." << std::endl;
        exit(0);
    }
    if (blockGridX > maxGridDim.x || blockGridY > maxGridDim.y || blockGridZ > maxGridDim.z) {
        std::cout << "Grid contain more thread blocks than the max value." << std::endl;
        exit(0);
    }
    m_gridSize = { blockGridX, blockGridY, blockGridZ };
    m_threadBlockSize = { threadBlockX, threadBlockY, threadBlockZ };
    m_sharedMemorySize = optimalThreadsPerBlock * (3 + 1) * sizeof(float);    // 3 pos, 1 mass
}
