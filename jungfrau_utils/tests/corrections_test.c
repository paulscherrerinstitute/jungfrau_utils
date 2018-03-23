#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdint.h>
// i 2 MSB possono essere 00 01 11 (10 e' escluso)
#include <time.h>

void pseudo_C(uint16_t m, uint16_t n, uint16_t *image, float *G, float *P, float *res)
{
  uint16_t gm;

  for(int i=0; i < m; i++)
    for(int j=0; j < n; j++){
      uint32_t idx = i * n + j;
      gm = image[idx] >> 14; // 1100000000000000 in hex
      //printf("%d\n", i);
      res[idx] = ((image[idx] & 0x3FFF) - P[(gm - 1) * m * n + i * n + j]) * (G[(gm - 1) * m * n + i * n + j]);
    }
}


void baseline(uint16_t m, uint16_t n, uint16_t *image, float *G, float *P, float *res){
  uint16_t gm;
  uint32_t idx_t, idx;
  for(int i=0; i < m; i++){
    idx_t = i * n;
    for(int j=0; j < n; j++){
      idx = idx_t + j;
      //gm = image[idx] >> 14; // 1100000000000000 in hex
      //printf("%d\n", i);
      res[idx] = ((image[idx] /*& 0x3FFF*/) - P[idx]) * (G[idx]);
    }
  }
}


int main(){
/*
  uint16_t size_x = 4096;
  uint16_t size_y = 4096;
*/

  uint16_t size_x = 1024;
  uint16_t size_y = 1024;

  uint16_t *image; //[size_x * size_y];
  float *G, *P, *res;

  image = (uint16_t*) malloc(size_x * size_y * sizeof(uint16_t));
  G = (float *) malloc(3 * size_x * size_y * sizeof(float));
  P = (float *) malloc(3 * size_x * size_y * sizeof(float));
  res = (float *) malloc(size_x * size_y * sizeof(float));
  //float P[3][size_x][size_y];
  //float res[size_x][size_y];
 
  
  for(int i=0; i<size_x; i++)
    for(int j=0; j<size_y; j++){
      //printf("%d\n", i);
      
      image[i * size_y + j] = rand();
      res[i * size_y + j] = 0;
	
  	  for(int z=0; z<3; z++){
        G[z * size_x * size_y + i * size_y + j] = rand();
	      P[z * size_x * size_y + i * size_y + j] = rand();
	    } 
	
    }

clock_t start = clock(), diff;
  
for(int i=0; i<100;i++)
  pseudo_C(size_x, size_y, image, G, P, res);
  //baseline(size_x, size_y, image, G, P, res);

diff = clock() - start;

int msec = diff * 1000 / CLOCKS_PER_SEC;
printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);  
  return 0;
}
