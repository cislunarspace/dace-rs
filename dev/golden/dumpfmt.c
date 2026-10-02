/* Print a few DAs through daceWrite for exact-format comparison. */
#include <stdio.h>
#include "dace/dacebase.h"

int main(void)
{
    daceInitialize(3, 2);
    DACEDA x, y, f, z;
    daceAllocateDA(&x, 0); daceAllocateDA(&y, 0); daceAllocateDA(&f, 0); daceAllocateDA(&z, 0);
    daceCreateVariable(&x, 1, 1.0);
    daceCreateVariable(&y, 2, 1.0);
    daceCreateConstant(&f, 1.0);
    daceAddDouble(&f, 0.0, &f);
    DACEDA t1, t2;
    daceAllocateDA(&t1, 0); daceAllocateDA(&t2, 0);
    daceMultiplyDouble(&x, 2.0, &t1);
    daceAdd(&f, &t1, &f);
    daceMultiply(&y, &y, &t2);
    daceMultiplyDouble(&t2, -0.5, &t2);
    daceAdd(&f, &t2, &f);
    unsigned int n = 0;
    char buf[140 * 64];
    daceWrite(&f, buf, &n);
    for (unsigned int i = 0; i < n; i++)
        printf("[%s]\n", &buf[i * 140]);
    daceWrite(&z, buf, &n);
    for (unsigned int i = 0; i < n; i++)
        printf("[%s]\n", &buf[i * 140]);
    return 0;
}
