#include <mex.h>
#include <matrix.h>
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

//declare variables
    mxArray *a_in_m, *b_in_m;
    mxArray *c_out_m, *d_out_m;
    const mwSize *dimsSignal;
    double *a, *b, *c, *d;
    int dimx, dimy, numdims;

//associate inputs
    a_in_m = mxDuplicateArray(prhs[0]);
    b_in_m = mxDuplicateArray(prhs[1]);

//figure out dimensions
    dimsSignal = mxGetDimensions(prhs[0]);
    numdims = mxGetNumberOfDimensions(prhs[0]);
    dimy = (int)dimsSignal[0]; dimx = (int)dimsSignal[1];

//associate outputs
    c_out_m = plhs[0] = mxCreateDoubleMatrix(dimy,dimx,mxREAL);
    d_out_m = plhs[1] = mxCreateDoubleMatrix(dimy,dimx,mxREAL);

//associate pointers
    a = mxGetPr(a_in_m);
    b = mxGetPr(b_in_m);
    c = mxGetPr(c_out_m);
    d = mxGetPr(d_out_m);

//do something
    double s;
    int v[dimx];
    float z[dimx+1];
    int k=0;
    v[0]=1;
    z[0]=LONG_MIN;
    int i=2;
    while (i<=dimx)
    {
        // compute intersection of parabola from i with parabola from v[k]
        s=(-b[0]*(i-v[k])*(i+v[k])+a[i-1]-a[(v[k])-1]+b[1]*(i-v[k]))/(-2*b[0]*(i-v[k]));
        if (s>z[k]) // Intersection is to the right of the previous change pt.
        {    
            k=k+1;  // Add current parabola to envelope
            if (k>dimx) mexErrMsgTxt("k exceeds dimx"); 
            z[k]=s; // Update rightmost intersection point
            v[k]=i; // Update right most parabola index
            i=i+1;  // Increment parabola index
        }
        else k=k-1;  // Decrement k to compare with previous parabola in envelope
    }
    z[k+1]=LONG_MAX;     // Last parabola is minimal till infinity
    k=0;
    for (i=1;i<=dimx;i++)
    {
        while (z[k+1]<i)
        {
            k=k+1;  // compute the parabola index for given domain point
        }
        //Get score and position
        c[i-1]=-(b[0]*(i-v[k])*(i-v[k])+b[1]*(i-v[k]))+a[(v[k])-1];
        d[i-1]=v[k];
    }

return;
}