
double fv(double r, int nper, double pv){
   double fv = (pv * pow((1 + r),nper)) - pv;
	return fv;
}


double getNCR( double MIDR, double * N){

double ROIt1=.166;
double ROItn=.135;
double GRWt1=.228;
double GRWtn=.119;
double DRt1=MIDR;
double HGR=.2161;
double GIt0=20;
double NDA_pct=.513;
double HZN=5;
double RT=0;
double ROIFT=.06;
double GRWFT=.025;
double DRFT=.06;
double life=7;

ROIt1=N[1];
ROItn=N[2];
GRWt1=N[3];
GRWtn=N[4];
//MIDR 
HGR  =N[6];
GIt0 =N[7];
NDA_pct=N[8];
HZN=N[9];
RT=N[10];
ROIFT=N[11];
GRWFT=N[12];
DRFT=N[13];
life=N[14];



HZN=(int)HZN;
double DAt0=GIt0*NDA_pct;

int PER[141];
for(int j=0; j<=141; j++){
PER[j]=j;
}


double ROI[141];
RT=(ROItn-ROIt1)/(HZN-1);
ROI[0]=0;
ROI[1]=ROIt1;
for(int j=2; j<=HZN; j++){
ROI[j]=ROI[j-1]+RT;
}
for(int j=HZN+1; j<=141; j++){
ROI[j]=ROI[j-1]+((ROIFT-ROI[j-1])/10);
}


double GRW[141];
GRW[0]=0;
GRW[1]=GRWt1;
RT=(GRWtn-GRWt1)/(HZN-1);
for(int j=2; j<=HZN; j++){
GRW[j]=GRW[j-1]+RT;
}
for(int j=HZN+1; j<=141; j++){
GRW[j]=GRW[j-1]+((GRWFT-GRW[j-1])/10);
}

double GI[141];
GI[0]=GIt0;
for(int j=1; j<=HZN; j++){
GI[j]=GI[j-1]*(1+GRW[j]);
}
for(int j=HZN+1; j<=141; j++){
GI[j]=GI[j-1]*(1+GRW[j]);
}

double DA[141];
DA[0]=DAt0;
for(int j=1; j<=141; j++){
DA[j]=DA[j-1]+(DA[j-1]*GRW[j]);
}

double NDA[141];
for(int j=0; j<=141; j++){
NDA[j]=GI[j]-DA[j];
}

double GCF[141];
GCF[0]=0;
for(int j=1; j<=141; j++){
//PMT ROI Y GI  
//GCF[j]=ROI[j]*GI[j];
GCF[j]=fv(ROI[j],life,GI[j]);
}

double DR[141];
DR[0]=0;
for(int j=1; j<=HZN; j++){
DR[j]=DRt1;
}
for(int j=HZN+1; j<=141; j++){
DR[j]=DR[j-1]+((DRFT-DR[j-1])/10);
}


double PVF[141];
PVF[0]=0;
for(int j=1; j<=141; j++){
PVF[j]=1/pow(1+DR[j],j);
}

double INV[141];
INV[0]=0;
for (int j=1; j<=141; j++){
INV[j]=GI[j]-GI[j-1];
}

double MINV[141];
MINV[0]=0;
for (int j=1; j<=141; j++){
//lookback post life 
//within life assume payments over prior life to get to DA @ 0 
MINV[j]=DAt0/5;
}

double TINV[141];
for(int j=0; j<=141; j++){
TINV[j]=INV[j]+MINV[j];
}

double PVGCF[141];
for(int j=0; j<=141; j++){
PVGCF[j]=GCF[j]*PVF[j];
}

double PVTINV[141];
for(int j=0; j<=141; j++){
PVTINV[j]=TINV[j]*PVF[j];
}

double NCR[141];
for(int j=0; j<=141; j++){
NCR[j]=PVGCF[j]-PVTINV[j];
}

double total_NCR=0;
for(int j=0; j<=141; j++){
total_NCR=total_NCR+NCR[j];
}

return total_NCR;
}


__kernel void vector_add(__global double *A, __global double *B, __global double *C, __global double *N) {
    
    // Get the index of the current element
    int i = get_global_id(0);
   

    // Do the operation
	A[i]=A[i];
	B[i]=B[i];
	double target=N[i*15];
	double guess= B[i];
	
	double args[15];
	for (int q=0; q<=15; q++)
	{
	args[q]=N[q];
	}
	
		
	//5 tries to get to the target NCR 
	for (int z=0; z<=20; z++){
		double result=getNCR(guess,args);
		double pct_dif=(target-result)/target;
		guess=guess*(1-pct_dif);
	}

	//Return the MIDR and Seeked NCR 
	A[i]=target;
	B[i]=guess;
	C[i]=getNCR(guess,args);
	
	
}
