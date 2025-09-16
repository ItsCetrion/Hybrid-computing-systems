void addVectorsCPU(const float *h_a, const float *h_b, float *h_result, const int n){
  for (int i = 0; i < n; i++){
    h_result[i] = h_a[i] + h_b[i];
  }
}