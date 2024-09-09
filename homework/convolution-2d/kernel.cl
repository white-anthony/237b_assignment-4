__kernel void convolution2D(
    __global float *inputData,     
    __global float *outputData,     
    __constant float *maskData,    
    int width,                      
    int height,                    
    int maskWidth,                  
    int imageChannels               
) {
    int maskRadius = maskWidth / 2;  

       int i = get_global_id(1);  // Row index
    int j = get_global_id(0);  // Column index

        if (i < height && j < width) {
        // Loop over each color channel
        for (int k = 0; k < imageChannels; k++) {
            float accum = 0.0f;

                       for (int y = -maskRadius; y <= maskRadius; y++) {
                for (int x = -maskRadius; x <= maskRadius; x++) {
                    int xOffset = j + x;
                    int yOffset = i + y;

                                       if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height) {
                        // Compute the index for the input pixel and mask value
                        float imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                        float maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];
                        accum += imagePixel * maskValue;
                    }
                }
            }

            outputData[(i * width + j) * imageChannels + k] = clamp(accum, 0.0f, 1.0f);
        }
    }
}
