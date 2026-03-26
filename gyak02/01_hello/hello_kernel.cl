__kernel void hello_kernel(__global int* buffer, int n) {
    int id = get_global_id(0);
    if (id < n) {
        buffer[id] = 11;
    }
}