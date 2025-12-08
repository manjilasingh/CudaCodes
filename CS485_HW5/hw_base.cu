/*
Name:Md Kamal Hossain Chowdhury
Email: mhchowdhury@crimson.ua.edu 
Course: CS 691
Homework #: 1

*/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define DIES 0
#define ALIVE 1
#define blockSize 256

const int TILE_DIM = 32;



/* function to measure time taken */
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void printarray(int *a, int M, int N, FILE *fp) {
  int i, j;
  for (i = 0; i < M+2; i++) {
    for (j = 0; j< N+2; j++)
      fprintf(fp, "%d ", a[i*(N+2) + j]);
    fprintf(fp, "\n");
  }
}

int check_array(int *a, int M, int N) {
  int value=0;
  for (int i = 1; i < M+1; i++)
    for (int j = 1; j< N+1; j++)
      value+= a[i*(N+2) + j];
  return value;
}
int compare_array(int *a,int *b, int M, int N) {
  int flag=1;
  for (int i = 1; i < M+1; i++)
    for (int j = 1; j< N+1; j++)
      if(a[i*(N+2) + j]!=b[i*(N+2) + j])
        {

          printf("Failed life[%d][%d]=%d h_life[%d][%d]=%d\n",i,j,a[i*(N+2) + j],i,j,b[i*(N+2) + j]);
          flag= 0;
          return flag;
        }
  return flag;
}


__global__
void compute_gpu_stride(int *life, int *temp, int M, int N) {
  // int  value;
  int index_x = blockIdx.x * blockDim.x + threadIdx.x+1;
  // int index_y = blockIdx.y * blockDim.y + threadIdx.y+1;
 
  int strid=blockDim.x*gridDim.x;
  
  
  int neighbors;
   
    for (int i = 1; i <N+1 ; i++){
        for(int j=index_x ;j<N+1; j+=strid){
         int id=i*(N+2)+j;
         neighbors = life[id + (N + 2)] +                           // Upper neighbor
                    life[id - (N + 2)] +                           // Lower neighbor
                    life[id + 1] +                                      // Right neighbor
                    life[id - 1] +                                      // Left neighbor
                    life[id + (N + 3)] + life[id - (N + 3)] + // Diagonal neighbors
                    life[id - (N + 1)] + life[id + (N + 1)];


        temp[id] = (neighbors == 3 || (neighbors == 2 && life[id]))? 1 : 0;
        }
        }

  
 
   }

__global__ 
void compute_shared_gpu(int *life, int *temp, int M, int N)
{
  int neighbors=0;
  


   
	int col = (blockDim.x - 2) * blockIdx.x + threadIdx.x;
	int row = (blockDim.y - 2) * blockIdx.y + threadIdx.y; 	

  int my_id= (row * (N+2) + col);
  int shared_id= (threadIdx.x * blockDim.y + threadIdx.y);
		
	int shared_size_x = blockDim.y;
	__shared__ int tile[TILE_DIM* TILE_DIM+1];
    //extern __shared__ TYPE sh_lattice[];

 	if (col < N+2 && row < N+2) {
        tile[shared_id] = life[my_id];
 	}
    __syncthreads();

    // CHECK IF
	/*if (col < size_i+neighs && row < size_j+neighs && 
		threadIdx.x >= (neighs-1) && threadIdx.x < blockDim.x-neighs && 
		threadIdx.y >= (neighs-1) && threadIdx.y < blockDim.y-neighs) {*/
    
    if (col < N+1 && row < N+1 && 
		threadIdx.x >= 1 && threadIdx.x < blockDim.x-1 && 
		threadIdx.y >= 1 && threadIdx.y < blockDim.y-1) {    
        
    //neighbors = neighbors_neighs(shared_id, shared_size_x-halo, sh_lattice, neighs, halo);	// decrease shared_size_x by 2 to use the same neighbors_neighs function than the rest of the implementations
    neighbors =  tile[shared_id - shared_size_x - 1];
    neighbors += tile[shared_id - shared_size_x];
    neighbors += tile[shared_id - shared_size_x + 1];
    neighbors += tile[shared_id - 1];
    neighbors += tile[shared_id + 1];
    neighbors += tile[shared_id + shared_size_x - 1];
    neighbors += tile[shared_id + shared_size_x];
    neighbors += tile[shared_id + shared_size_x + 1];

    temp[my_id] = (neighbors == 3 || (neighbors == 2 && life[my_id]))? 1 : 0;

    //check_rules(my_id, neighbors, d_lattice, d_lattice_new);
 	}
}
  
__global__
void compute_gpu(int *life, int *temp, int M, int N) {
  
     
  int x = blockIdx.x * blockDim.x + threadIdx.x+1;
  int y = blockIdx.y * blockDim.y + threadIdx.y+1;
    
  int id= x*(N+2)+y;
  int neighbors;
  if(x<=N &&y<=N){
      neighbors = life[id + (N + 2)] +                           // Upper neighbor
                    life[id - (N + 2)] +                           // Lower neighbor
                    life[id + 1] +                                      // Right neighbor
                    life[id - 1] +                                      // Left neighbor
                    life[id + (N + 3)] + life[id - (N + 3)] + // Diagonal neighbors
                    life[id - (N + 1)] + life[id + (N + 1)];

        temp[id] = (neighbors == 3 || (neighbors == 2 && life[id]))? 1 : 0;
    }
    

}

   


void compute(int *life, int *temp, int M, int N) {
  int i, j, value;

  for (i = 1; i < M+1; i++) {
    for (j = 1; j < N+1; j++) {
      /* find out the value of the current cell */
      value = life[(i-1)*(N+2) + (j-1)] + life[(i-1)*(N+2) + j] + 
              life[(i-1)*(N+2) + (j+1)] + life[i*(N+2) + (j-1)] + 
              life[i*(N+2) + (j+1)] + life[(i+1)*(N+2) + (j-1)] + 
              life[(i+1)*(N+2) + j] + life[(i+1)*(N+2) + (j+1)] ;
     
      
      /* check if the cell dies or life is born */
      if (life[i*(N+2) + j]) { // cell was alive in the earlier iteration
	if (value < 2 || value > 3) {
	  temp[i*(N+2) + j] = DIES ;
	}
	else // value must be 2 or 3, so no need to check explicitly
	  temp[i*(N+2) + j] = ALIVE ; // no change
      } 
      else { // cell was dead in the earlier iteration
	if (value == 3) {
	  temp[i*(N+2) + j] = ALIVE;
	}
	else
	  temp[i*(N+2) + j] = DIES; // no change
      }
    }
  }

}


int main(int argc, char **argv) {
  int N, NTIMES, *life=NULL, *temp=NULL,*d_life=NULL,*d_temp=NULL,*h_life=NULL,*h_temp=NULL;
  int i, j, k;
  

  //int *life_stride=NULL,*temp_stride=NULL;
  // double t1_gpu,t2_gpu;
  
#if defined(DEBUG1) || defined(DEBUG2)
  FILE *fp;
  char filename[32];
#endif
if (argc != 4) {
        printf("Usage: %s <board size> <max number of generations> <directory for output file>\n", argv[0]);
        return -1;
    }
  N = atoi(argv[1]);
  NTIMES = atoi(argv[2]);
  char *directory = argv[3];

  /* Allocate memory for both arrays */
  life = (int *)malloc((N+2)*(N+2)*sizeof(int));
  temp = (int *)malloc((N+2)*(N+2)*sizeof(int));
  //life_stride = (int *)malloc((N+2)*(N+2)*sizeof(int));
  //temp_stride = (int *)malloc((N+2)*(N+2)*sizeof(int));

  /* Initialize the boundaries of the life matrix */
  for (i = 0; i < N+2; i++) {
    life[i*(N+2)] = life[i*(N+2) + (N+1)] = DIES ;
    temp[i*(N+2)] = temp[i*(N+2) + (N+1)] = DIES ;
  }
  for (j = 0; j < N+2; j++) {
    life[j] = life[(N+1)*(N+2) + j] = DIES ;
    temp[j] = temp[(N+1)*(N+2) + j] = DIES ;
  }

  /* Initialize the life matrix */
  for (i = 1; i < N+1; i++) {
    for (j = 1; j< N+1; j++) {
      srand(54321|i);
      if (drand48() < 0.5) 
	      life[i*(N+2) + j] = ALIVE ;
      else
	      life[i*(N+2) + j] = DIES ;
    }
  }
   //life_stride=life;  //copy life for stride kernel use
   //temp_stride=temp; //copy temp for stride kernel use

//gpu programming
  // int numBlocks = (N + blockSize - 1) / blockSize;
  // dim3 dimGrid_strid(numBlocks,numBlocks,1);
 
 
  //int blockSizeShared=(N+(TILE_DIM-2)-1)/(TILE_DIM-2);
  // dim3 dimGrid(blockSizeShared, blockSizeShared, 1);
  // dim3 dimBlock(TILE_DIM, TILE_DIM, 1);

  
  // dim3 dimGrid((N+2+(TILE_DIM-2)-1)/(TILE_DIM-2), (N+2+(TILE_DIM-2)-1)/(TILE_DIM-2), 1);
  // dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  dim3 dimBlock(8, 8);
  dim3 dimGrid((N + dimBlock.x -1) / dimBlock.x, (N+dimBlock.y -1) / dimBlock.y);
  fprintf(stderr,"dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
  cudaMalloc(&d_life, (N+2)*(N+2)*sizeof(int)); 
  cudaMalloc(&d_temp, (N+2)*(N+2)*sizeof(int));
  h_life = (int *)malloc((N+2)*(N+2)*sizeof(int));
  h_temp = (int *)malloc((N+2)*(N+2)*sizeof(int));
  
  cudaMemcpy(d_life, life, (N+2)*(N+2)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_temp, temp, (N+2)*(N+2)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h_life, d_life, (N+2)*(N+2)*sizeof(int), cudaMemcpyDeviceToHost); 

  
#ifdef DEBUG1
  /* Display the initialized life matrix */
  fprintf(stderr,"Printing to file: gpuoutput.%d.0\n",N);
  sprintf(filename,"gpuoutput.%d.0",N);
  fp = fopen(filename, "w");
  printarray(life, N, N, fp);
  fprintf(fp,"\n-----------\n");
  printarray(h_life, N, N, fp);
  fclose(fp);
#endif
// events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;
  checkCuda( cudaEventRecord(startEvent, 0) );

  for (k = 0; k < NTIMES; k += 2) {

    // compute_shared_gpu<<<dimGrid,dimBlock>>>(d_life,d_temp,N,N);
    // compute_shared_gpu<<<dimGrid,dimBlock>>>(d_temp,d_life,N,N);
    compute_gpu<<<dimGrid,dimBlock>>>(d_life,d_temp,N,N);
    compute_gpu<<<dimGrid,dimBlock>>>(d_temp,d_life,N,N);
  }

  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  fprintf(stderr,"----------------------------------\n");
  fprintf(stderr,"Baseline  GPU time is taken=%f ms for size=%d iterations=%d\n",ms,N,NTIMES);
    
  cudaMemcpy(h_life, d_life, (N+2)*(N+2)*sizeof(int), cudaMemcpyDeviceToHost); 
  cudaMemcpy(h_temp, d_temp, (N+2)*(N+2)*sizeof(int), cudaMemcpyDeviceToHost); 
  

#ifdef DEBUG1
  /* Display the life matrix after k iterations */
  printf("Printing to file: output.%d.%d\n",N,k);
  sprintf(filename,"gpuoutput.%d.%d",N,k);
  fp = fopen(filename, "w");
  printarray(h_temp, N, N, fp);
  fprintf(fp, "\n--------------------\n");
  printarray(h_life,N,N,fp);
  fclose(fp);
#endif

    FILE *f;
    f = fopen(directory, "w");
    if(f == NULL){
        printf("Error opening output file\n");
        return -1;
    }
    for(int i = 1; i < N+1; i++){
        for(int j = 1; j < N+1; j++){
            fprintf(f, " %d ", h_life[i*(N+2) + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

  cudaFree(d_life);
  cudaFree(d_temp);
  free(h_life); 
  free(h_temp);
  free(life);
  free(temp);
  return 0;
}