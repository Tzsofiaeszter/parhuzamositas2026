// fill_gaps.cl
// Hiányzó elemek pótlása szomszédos átlaggal
__kernel void fill_gaps(__global int* data, const int n) {
    int gid = get_global_id(0);

    // csak akkor dolgozunk, ha elem 0 (hiányzó)
    if (gid > 0 && gid < n-1 && data[gid] == 0) {
        int left = data[gid-1];
        int right = data[gid+1];
        data[gid] = (left + right) / 2;
    }
}