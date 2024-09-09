__kernel void convolution2D(__global float *inputData, __global float *outputData,
                           __constant float *maskData, const unsigned int width,
                           const unsigned int height, const unsigned int maskWidth,
                           const unsigned int imageChannels) {
    const unsigned int maskRadius = maskWidth / 2;

    int x = get_global_id(0);
    int y = get_global_id(1);

    for (int c = 0; c < imageChannels; ++c) {
        float accum = 0.0f;

        int startX = max((int)0, (int)(x - maskRadius));
        int startY = max((int)0, (int)(y - maskRadius));
        int endX = min((int)width - 1, (int)(x + maskRadius));
        int endY = min((int)height - 1, (int)(y + maskRadius));

        for (int ky = startY; ky <= endY; ++ky) {
            for (int kx = startX; kx <= endX; ++kx) {
                int inputIndex = (ky * width + kx) * imageChannels + c;
                int maskIndex = (ky - startY + maskRadius) * maskWidth + (kx - startX + maskRadius);
                float imagePixel = inputData[inputIndex];
                float maskValue = maskData[maskIndex];

                accum += imagePixel * maskValue;
            }
        }

        outputData[(y * width + x) * imageChannels + c] = clamp(accum, 0.0f, 1.0f);
    }
}