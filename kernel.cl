__kernel void convolution3x3(__global float *input,__global float *output,__global float *filter, int N,int D1,int D2,__global float *bias) {
	int i=get_global_id(0);
	int j=get_global_id(1);
	int filter_i=27*(j);
	int row=i/N%N;
	int col=i%N;
	float sum=0;

	if(row-1>=0&&col-1>=0){	
		sum+=input[N*(row-1)+(col-1)]*filter[filter_i];
		sum+=input[N*(row-1)+(col-1)+N*N]*filter[filter_i+9];
		sum+=input[N*(row-1)+(col-1)+2*N*N]*filter[filter_i+18];
	}

	if(row-1>=0){
		sum+=input[N*(row-1)+col]*filter[filter_i+1];
		sum+=input[N*(row-1)+col+N*N]*filter[filter_i+1+9];
		sum+=input[N*(row-1)+col+2*N*N]*filter[filter_i+1+18];
	}

	if(row-1>=0&&col+1<N){ 
		sum+=input[N*(row-1)+col+1]*filter[filter_i+2];
		sum+=input[N*(row-1)+col+1+N*N]*filter[filter_i+2+9];
		sum+=input[N*(row-1)+col+1+2*N*N]*filter[filter_i+2+18];
	}
	
	if(col-1>=0){ 
		sum+=input[N*row+col-1]*filter[filter_i+3];
		sum+=input[N*row+col-1+N*N]*filter[filter_i+3+9];
		sum+=input[N*row+col-1+2*N*N]*filter[filter_i+3+18];
	}

	sum+=input[N*row+col]*filter[filter_i+4];
	sum+=input[N*row+col+N*N]*filter[filter_i+4+9];
	sum+=input[N*row+col+2*N*N]*filter[filter_i+4+18];

	if(col+1<N){ 
		sum+=input[N*row+col+1]*filter[filter_i+5];
		sum+=input[N*row+col+1+N*N]*filter[filter_i+5+9];
		sum+=input[N*row+col+1+2*N*N]*filter[filter_i+5+18];
	}
	if(row+1<N&&col-1>=0){ 
		sum+=input[N*(row+1)+col-1]*filter[filter_i+6];
		sum+=input[N*(row+1)+col-1+N*N]*filter[filter_i+6+9];
		sum+=input[N*(row+1)+col-1+2*N*N]*filter[filter_i+6+18];
	}
	if(row+1<N){ 
		sum+=input[N*(row+1)+col]*filter[filter_i+7];
		sum+=input[N*(row+1)+col+N*N]*filter[filter_i+7+9];
		sum+=input[N*(row+1)+col+2*N*N]*filter[filter_i+7+18];
	}
	if(row+1<N&&col+1<N){ 
		sum+=input[N*(row+1)+col+1]*filter[filter_i+8];
		sum+=input[N*(row+1)+col+1+N*N]*filter[filter_i+8+9];
		sum+=input[N*(row+1)+col+1+2*N*N]*filter[filter_i+8+18];
	}
	


	output[i+(N*N)*j]=(sum+bias[j])>0?sum+bias[j]:0;

}