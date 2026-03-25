__kernel void sample_kernel(__global int* buffer, int n)
{
    int i = 0;
    for (;i < get_global_id(0); i++) {
        if (i < n) {
            buffer[i] = i;
        }
    }
}
