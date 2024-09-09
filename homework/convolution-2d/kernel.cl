// Define min and max functions
inline int min(int a, int b) {
    return a < b ? a : b;
}

inline int max(int a, int b) {
    return a > b ? a : b;
}

__kernel void convolution2D(__global float *inputData, __global float *outputData, __constant float *maskData, 
                             const unsigned int width, const unsigned int height, const unsigned int maskWidth, const unsigned int imageChannels)
{
    __local float sharedMem[16 * 16 * 3];  

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    // Load data into shared memory
    if (x < width && y < height)
    {
        int sharedIndex = (ty * get_local_size(0) + tx) * imageChannels;
        int globalIndex = (y * width + x) * imageChannels;

        for (int c = 0; c < imageChannels; ++c)
        {
            sharedMem[sharedIndex + c] = inputData[globalIndex + c];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform convolution
    if (x < width && y < height)
    {
        float result = 0.0f;

        for (int ky = -2; ky <= 2; ++ky)
        {
            for (int kx = -2; kx <= 2; ++kx)
            {
                int imgX = min(max(x + kx, 0), width - 1);
                int imgY = min(max(y + ky, 0), height - 1);

                float maskValue = maskData[(ky + 2) * maskWidth + (kx + 2)];
                for (int c = 0; c < imageChannels; ++c)
                {
                    result += sharedMem[((ty + ky) * get_local_size(0) + (tx + kx)) * imageChannels + c] * maskValue;
                }
            }
        }

        result = clamp(result, 0.0f, 1.0f);
        outputData[(y * width + x) * imageChannels] = result;
    }
}
