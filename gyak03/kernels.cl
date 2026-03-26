// kernels.cl
// Visszafelé feltöltés és páros-páratlan csere

__kernel void map_reverse_and_swap(__global int* data, const int n) {
    int gid = get_global_id(0);

    // 1) Visszafelé töltés
    if (gid < n) {
        data[gid] = n - gid - 1;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // 2) Páros-páratlan csere
    int i = gid * 2;
    if ((i + 1) < n) {
        int temp = data[i];
        data[i] = data[i + 1];
        data[i + 1] = temp;
    }
}