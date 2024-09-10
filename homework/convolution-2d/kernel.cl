__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth,  int imageChannels){
    int maskRadius = maskWidth / 2;

    int j = get_global_id(0);  
    int i = get_global_id(1); 

    if (i < height && j < width) {
        for (int k = 0; k < imageChannels; k++) {
            float accum = 0.0f;  
            for (int y = -maskRadius; y <= maskRadius; y++) {
                for (int x = -maskRadius; x <= maskRadius; x++) {
                    // Calculate offsets
                    int xOffset = j + x;
                    int yOffset = i + y;

                    if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height) {
                        float imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                        float maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];
                        
                        accum += imagePixel * maskValue;
                    }
                }
            }

            accum = clamp(accum, 0.0f, 1.0f);

            outputData[(i * width + j) * imageChannels + k] = accum;
        }
    }
}
