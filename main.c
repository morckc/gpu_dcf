#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h> /* PRIuPTR and uintptr_t */

/* errno and strerror */
#include <errno.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* see check_opencl.h for docs on the CHECK_* macros */
#include "check_opencl.h"

#define MAX_SOURCE_SIZE (0x100000)

#define val_size 10000
char val[val_size];

char *sourcepath= "dcf_kernel.cl";

int main(void) {
    DECLARE_CHECK;

	// Create the two input vectors
    int i;
    const int LIST_SIZE = 25600;

	int *ID = CHECK_malloc(sizeof(int)*LIST_SIZE,err_malloc_ID); 
	double *A = CHECK_malloc(sizeof(double)*LIST_SIZE, err_malloc_A);
    double *B = CHECK_malloc(sizeof(double)*LIST_SIZE, err_malloc_B);
	int arg=15;
	double INPUTS[15]= {25000,.06,.06,.025,.025,.06,.025,2000,.4,5.0,0.00,.06,.025,.06,7};
	double *N = CHECK_malloc(sizeof(double)*(LIST_SIZE*arg+arg),err_malloc_B);

	int RandRange(double Min, double Max)
	{
    int diff = Max-Min;
    return (double) (((double)(diff+1)/RAND_MAX) * rand() + Min);
	}

	double INPUTS_RANDOMIZED[15];
   
	for(i = 0; i <= LIST_SIZE; i++) {
	ID[i]=i;
	memcpy(INPUTS_RANDOMIZED,INPUTS,sizeof(INPUTS));
	
	//.5 B to 500B  Sum NCR 
	INPUTS_RANDOMIZED[0]= RandRange(5,25000);

	// 6% ROI 2.5% RAGR  requiring  12.5X  Sum NCR/GI to get 6% MIDR 
	// something is up.... but continuing on 
	// 7 to 20x  NCR/GI
	double mult = 12.5; 
	INPUTS_RANDOMIZED[7]=INPUTS_RANDOMIZED[0]/mult;
	
	INPUTS_RANDOMIZED[1]=.06;					//ROI 6%
	INPUTS_RANDOMIZED[2]=INPUTS_RANDOMIZED[1];  //hold flat till t5 
	INPUTS_RANDOMIZED[3]=.025;					//RAGR 2.5%
	INPUTS_RANDOMIZED[4]=INPUTS_RANDOMIZED[3];	//holt flat till t5 
	//HGR constant for everyone 
	INPUTS_RANDOMIZED[8]=.4;	//NDA% between 20 and 80%

	for(int q=0; q<= arg; q++){
		N[(i*arg)+q] = INPUTS_RANDOMIZED[q];
		}	
//	A[i] = INPUTS[0]+rand()%100;  //target NCR	
	B[i] = .06;				 //initial MIDR Guess 
	}

	struct RES {
		int ITER; 
		int ID[LIST_SIZE];
		double A[LIST_SIZE];
		double B[LIST_SIZE];
		double C[LIST_SIZE];
		double DIFF[LIST_SIZE];
		double N[LIST_SIZE*arg];
	};
   	
    // Load the kernel source code into the array source_str
    FILE *fp = fopen(sourcepath, "r");
    if (!fp) {
	fprintf(stderr, "Failed to open kernel file '%s': %s\n", sourcepath,
		strerror(errno));
	inc_CHECK_errors();
	goto err_fopen;
    }

    char *source_str = CHECK_malloc(MAX_SOURCE_SIZE, err_malloc_source_str);
    size_t source_size= fread( source_str, 1, MAX_SOURCE_SIZE, fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    CHECK_clGetPlatformIDs(1, &platform_id, &ret_num_platforms,
			   err_clGetPlatformIDs);

    fprintf(stderr, "ret_num_platforms=%i\n", ret_num_platforms);

    CHECK_clGetDeviceIDs( platform_id,
			  CL_DEVICE_TYPE_GPU,
			  1,
			  &device_id,
			  &ret_num_devices,
			  err_clGetDeviceIDs);

    // Create an OpenCL context
    cl_context context =
	CHECK_clCreateContext( NULL, 1, &device_id, NULL, NULL, err_clCreateContext);

    // Create a command queue
    cl_command_queue command_queue =
	CHECK_clCreateCommandQueue(context, device_id, 0, err_clCreateCommandQueue);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj =
	CHECK_clCreateBuffer(context, CL_MEM_READ_ONLY, 
			     LIST_SIZE * sizeof(double), NULL, err_a_mem_obj);
    cl_mem b_mem_obj =
	CHECK_clCreateBuffer(context, CL_MEM_READ_ONLY,
			     LIST_SIZE * sizeof(double), NULL, err_b_mem_obj);
    cl_mem c_mem_obj =
	CHECK_clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
			     LIST_SIZE * sizeof(double), NULL, err_c_mem_obj);

	//CM TODO replace err_c_mem with err_n_mem 
    cl_mem n_mem_obj =
	CHECK_clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE*sizeof(double)*arg, NULL, err_c_mem_obj);


    // Copy the lists A and B to their respective memory buffers
    CHECK_clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
			       LIST_SIZE * sizeof(double), A, 0, NULL, NULL,
			       err_clEnqueueWriteBuffer_A);
    CHECK_clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
			       LIST_SIZE * sizeof(double), B, 0, NULL, NULL,
			       err_clEnqueueWriteBuffer_B);
	//CM TODO replace err_B with err_N
    CHECK_clEnqueueWriteBuffer(command_queue, n_mem_obj, CL_TRUE, 0, 
				  LIST_SIZE * sizeof(double)*arg, N, 0, NULL, NULL,
			       err_clEnqueueWriteBuffer_B);


    // Create a program from the kernel source
    cl_program program =
	CHECK_clCreateProgramWithSource(context,
					1,
					(const char**)&source_str,
					&source_size,
					err_clCreateProgramWithSource);

    free(source_str); 

    // Build the program
    cl_int ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret) {
	size_t sizeused;
	CHECK_clGetProgramBuildInfo (program,
				     device_id,
				     CL_PROGRAM_BUILD_LOG,
				     val_size-1, //?
				     &val,
				     &sizeused,
				     err_clGetProgramBuildInfo);

	printf("clBuildProgram error: (sizeused %"PRIuPTR") '%s'\n",
	       (uintptr_t) sizeused, val);
    err_clGetProgramBuildInfo:
	goto err_clBuildProgram;
    }

    // Create the OpenCL kernel
    cl_kernel kernel = CHECK_clCreateKernel(program, "vector_add", err_clCreateKernel);
    // Set the arguments of the kernel
    CHECK_clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj,err_clSetKernelArg);
    CHECK_clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem_obj,err_clSetKernelArg);
    CHECK_clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem_obj,err_clSetKernelArg);
	CHECK_clSetKernelArg(kernel, 3, sizeof(cl_mem), &n_mem_obj,err_clSetKernelArg);


    //Create a place to put C  
	double *C = CHECK_malloc(sizeof(double)*LIST_SIZE, err_malloc_C);
  
	//CPU LOOP  
	for (int j=0; j<1; j++){	
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 512; // Process in groups of 64
    CHECK_clEnqueueNDRangeKernel
	(command_queue, kernel, 1, NULL, 
	 &global_item_size, &local_item_size, 0, NULL, NULL,
	 err_clEnqueueNDRangeKernel);


    // Read the memory buffer C on the device to the local variable C
    C = CHECK_malloc(sizeof(double)*LIST_SIZE, err_malloc_C);
    CHECK_clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
			      LIST_SIZE * sizeof(double), C, 0, NULL, NULL,
			      err_clEnqueueReadBuffer);

    // Read the memory buffer A on the device to the local variable A (updating A)
    A = CHECK_malloc(sizeof(double)*LIST_SIZE, err_malloc_A);
    CHECK_clEnqueueReadBuffer(command_queue, a_mem_obj, CL_TRUE, 0, 
			      LIST_SIZE * sizeof(double), A, 0, NULL, NULL,
			      err_clEnqueueReadBuffer);

	// Read the memory buffer A on the device to the local variable B (updating B)
    B = CHECK_malloc(sizeof(double)*LIST_SIZE, err_malloc_B);
    CHECK_clEnqueueReadBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
			      LIST_SIZE * sizeof(double), B, 0, NULL, NULL,
			      err_clEnqueueReadBuffer);

	//DEBUG PRINT 
	int print = 0 ;	
	if(print == 1){ 
   	for(i = 0; i < LIST_SIZE; i++)
        printf("iter %d -- %d) %lf , %lf :  %lf \n", j, ID[i], A[i], B[i], C[i]);
	}

	struct RES r;
	r.ITER=j;	
	memcpy(r.ID,ID,sizeof(double)*LIST_SIZE);
	memcpy(r.A,A,sizeof(double)*LIST_SIZE);
	memcpy(r.B,B,sizeof(double)*LIST_SIZE);
	memcpy(r.C,C,sizeof(double)*LIST_SIZE);
    memcpy(r.N,N,sizeof(double)*LIST_SIZE);	
	for (i=0; i< LIST_SIZE; i++){
		r.DIFF[i]=(r.A[i]-r.C[i])/r.A[i];
	}	
	
	char fname[100];
	snprintf(fname,sizeof(fname),"output/file%d.dat",j);
	FILE *file;	
	file = fopen( fname , "w" );
	fwrite(&r, sizeof( r ), 1 ,file);
	fclose(file);
	}
	
	//debug read and write 	
	int print_file = 0; 
	if( print_file ==1 ){ 
	struct RES n; 
	FILE *file;
	file = fopen( "output/file0.dat" , "rb" );
	fread(&n, sizeof(n), 1,file);
	fclose(file);
	for( i =0; i< LIST_SIZE; i++)
    printf("iter %d -- %d) %lf  %lf %lf  %lf \n", n.ITER, n.ID[i], n.N[i], n.B[i], n.C[i],n.DIFF[i]);
	}	

err_clEnqueueReadBuffer:
    free(C);
 err_malloc_C:
    CHECK_clFlush(command_queue, err_clFlush);
 err_clFlush:
    CHECK_clFinish(command_queue, err_clEnqueueNDRangeKernel);
 err_clEnqueueNDRangeKernel:
 err_clSetKernelArg:
    CHECK_clReleaseKernel(kernel, err_clCreateKernel);
 err_clCreateKernel:
 err_clBuildProgram:
    CHECK_clReleaseProgram(program, err_clCreateProgramWithSource);
 err_clCreateProgramWithSource:
 err_clEnqueueWriteBuffer_B:
 err_clEnqueueWriteBuffer_A:
    CHECK_clReleaseMemObject(c_mem_obj, err_c_mem_obj);
 err_c_mem_obj:
    CHECK_clReleaseMemObject(b_mem_obj, err_b_mem_obj);
 err_b_mem_obj:
    CHECK_clReleaseMemObject(a_mem_obj, err_a_mem_obj);
 err_a_mem_obj:
    CHECK_clReleaseCommandQueue(command_queue, err_clCreateCommandQueue);
 err_clCreateCommandQueue:
    CHECK_clReleaseContext(context, err_clCreateContext);
 err_clCreateContext:
    // XXX deallocate device_id ?
 err_clGetDeviceIDs:
    // XXX deallocate platform_id ?
 err_clGetPlatformIDs:
 err_malloc_source_str:
    fclose( fp );
 err_fopen:
    free(B);
 err_malloc_B:
    free(A);
 err_malloc_ID:
	free(ID);
 err_malloc_A:
    return CHECK_errors;

}

