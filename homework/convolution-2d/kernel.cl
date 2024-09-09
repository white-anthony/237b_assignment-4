__kernel void convolution2D(__global float *inputData, __global float *outputData, __constant float *maskData, 
                             const unsigned int width, const unsigned int height, const unsigned int maskWidth, const unsigned int imageChannels)
{

    const unsigned int maskRadius = maskWidth / 2;


    int x = get_global_id(0);
    int y = get_global_id(1);


    for (int c = 0; c < imageChannels; ++c)
    {
        float accum = 0.0f;

        // Perform convolution
        for (int ky = -maskRadius; ky <= maskRadius; ++ky)
        {
            for (int kx = -maskRadius; kx <= maskRadius; ++kx)
            {
                int xOffset = x + kx;
                int yOffset = y + ky;

                // Check bounds
                if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height)
                {
                    int inputIndex = (yOffset * width + xOffset) * imageChannels + c;
                    int maskIndex = (ky + maskRadius) * maskWidth + (kx + maskRadius);
                    float imagePixel = inputData[inputIndex];
                    float maskValue = maskData[maskIndex];


                    accum += imagePixel * maskValue;
                }
            }
        }

        outputData[(y * width + x) * imageChannels + c] = clamp(accum, 0.0f, 1.0f);
    }
}

// Define clamp function
inline float clamp(float x, float lower, float upper)
{
    return min(max(x, lower), upper);
}
