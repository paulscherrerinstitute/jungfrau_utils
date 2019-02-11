#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// i 2 MSB possono essere 00 01 11 (10 e' escluso)
#include <time.h>

void pseudo_C(uint16_t m, uint16_t n, uint16_t *image, float *G, float *P, float *res) {
    uint16_t gm;
    uint32_t idx;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            idx = i * n + j;
            gm = image[idx] >> 14;  // 1100000000000000 in hex
            if (gm > 2) gm = 2;
            // printf("%d\n", i);
            res[idx] = ((image[idx] & 0x3FFF) - P[(gm)*m * n + idx]) / (G[(gm)*m * n + idx]);
        }
}

void jf_apply_pede_gain(uint32_t image_size, uint16_t *image, float *GP, float *res) {
    uint16_t gm;

    for (uint32_t idx = 0; idx < image_size; idx++) {
        gm = image[idx] >> 14;  // 1100000000000000 in hex
        if (gm == 3) gm = 2;
        //     R0 C0                   R0C1
        // g0 p0 g1 p1 g2 p2 g3 p3 | g0 p0 g1 p1 g2 p2 g3 p3
        res[idx] = ((image[idx] & 0x3FFF) - GP[8 * idx + 2 * gm + 1]) / GP[8 * idx + 2 * gm];
    }
}

void jf_apply_pede_gain_mask(uint32_t image_size, uint16_t *image, float *GP, float *res,
                             int *pixel_mask) {
    uint16_t gm;

    for (uint32_t idx = 0; idx < image_size; idx++) {
        if (pixel_mask[idx]) {
            res[idx] = 0;
        } else {
            gm = image[idx] >> 14;  // 1100000000000000 in hex
            if (gm == 3) gm = 2;
            //     R0 C0                   R0C1
            // g0 p0 g1 p1 g2 p2 g3 p3 | g0 p0 g1 p1 g2 p2 g3 p3
            res[idx] = ((image[idx] & 0x3FFF) - GP[8 * idx + 2 * gm + 1]) / GP[8 * idx + 2 * gm];
        }
    }
}

void baseline(uint16_t m, uint16_t n, uint16_t *image, float *G, float *P, float *res) {
    uint32_t idx_t, idx;

    for (int i = 0; i < m; i++) {
        idx_t = i * n;
        for (int j = 0; j < n; j++) {
            idx = idx_t + j;
            // printf("%d\n", i);
            image[idx] = image[idx] & 0x3FFF;
            // res[idx] = ((image[idx] & 0x3FFF) /*- P[idx]) / (G[idx]*/);
        }
    }
}

int main() {
    uint16_t size_x = 4096;
    uint16_t size_y = 4096;

    /*
    uint16_t size_x = 1024;
    uint16_t size_y = 1024;
  */
    uint16_t *image;  //[size_x * size_y];
    float *G, *P, *res, *GP;

    int N = 1000;

    image = (uint16_t *)malloc(size_x * size_y * sizeof(uint16_t));
    G = (float *)malloc(3 * size_x * size_y * sizeof(float));
    P = (float *)malloc(3 * size_x * size_y * sizeof(float));
    GP = (float *)malloc(8 * size_x * size_y * sizeof(float));

    res = (float *)malloc(size_x * size_y * sizeof(float));
    // float P[3][size_x][size_y];
    // float res[size_x][size_y];

    for (int i = 0; i < size_x; i++)
        for (int j = 0; j < size_y; j++) {
            // printf("%d\n", i);

            image[i * size_y + j] = rand();
            res[i * size_y + j] = 0;

            for (int z = 0; z < 3; z++) {
                G[z * size_x * size_y + i * size_y + j] = rand();
                P[z * size_x * size_y + i * size_y + j] = rand();
            }
            for (int z = 0; z < 4; z++) {
                GP[z * size_x * size_y + i * size_y + j] = rand();
                GP[z * size_x * size_y + i * size_y + j + 4] = rand();
            }
        }

    clock_t start = clock(), diff;

    int msec;

    // for(int i=0; i<N;i++)
    // res = pseudo_C2(size_x, size_y, image, GP);
    // pseudo_C(size_x, size_y, image, G, P, res);
    // baseline(size_x, size_y, image, G, P, res);

    diff = clock() - start;

    msec = diff * 1000 / CLOCKS_PER_SEC / N;
    printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

    start = clock();

    // for(int i=0; i<N;i++)
    //  baseline(size_x, size_y, image, G, P, res);

    diff = clock() - start;

    msec = diff * 1000 / CLOCKS_PER_SEC / N;
    printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

    return 0;
}
