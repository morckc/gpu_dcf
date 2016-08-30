#include <stdio.h>
#include <stdlib.h>
	
int main(int argi, char* argv[]){
	
int MAX_LIST_SIZE=25600;
struct RES {
        int ITER;
        int ID[MAX_LIST_SIZE];
        double A[MAX_LIST_SIZE];
        double B[MAX_LIST_SIZE];
        double C[MAX_LIST_SIZE];
 		double DIFF[MAX_LIST_SIZE]; 
		double N[MAX_LIST_SIZE*15];
	};

struct RES n; 
char fname[100];
int iter;
sscanf(argv[1],"%d",&iter);

snprintf(fname,sizeof(fname),"output/file%d.dat",iter);
FILE *fp = fopen( fname , "rb" );
    fread(&n, sizeof(n), 1,fp);
    fclose(fp);

/* 
	printf("Output for iteration #%d \n",n.ITER);
  	printf("Company TargetNCR   MIDR     SeekedNCR   Difference \n");
	for( int i =0; i< 10; i++){
		printf("comp_%d  %lf %lf %lf %lf %lf \n",  n.ID[i], n.A[i], n.B[i], n.C[i],n.DIFF[i],
		n.N[(i*15)+0]);
    }
*/
	
	char filename[10]="data.csv";
	FILE *file ;
	file=fopen(filename,"w+");
	fprintf(file,"id,targetncr,midr,seekedncr,diff,roit1,roitn,grwt1,grwtn,drt1,hgr,git0,ndapct,hzn,rt,roift,grwft,drft,life,x\n");
   	for( int i =0; i<MAX_LIST_SIZE; i++){
		fprintf(file,"comp_%d,%lf,%lf,%lf,%lf,",  n.ID[i], n.A[i], n.B[i], n.C[i],n.DIFF[i]);
		//Skipping 0 target NCR as we already have it from n.A	
		for (int q=1;q<15;q++){
				fprintf(file,"%lf,", n.N[(i*15)+q]);
		}
		fprintf(file,"X\n");
    }

}
