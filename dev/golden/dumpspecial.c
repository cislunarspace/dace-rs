/* Dump Bessel/gamma/psi results at (8,2) via daceWrite, sectioned by ====. */
#include <stdio.h>
#include "dace/dacebase.h"

static void dump(const DACEDA *d)
{
    char buf[140 * 8192];
    unsigned int n = 0;
    daceWrite(d, buf, &n);
    for (unsigned int i = 0; i < n; i++)
        printf("%s\n", &buf[i * 140]);
}

int main(void)
{
    daceInitialize(8, 2);
    DACEDA f, g;
    daceAllocateDA(&f, 0); daceAllocateDA(&g, 0);
    DACEDA x, y, t;
    daceAllocateDA(&x, 0); daceAllocateDA(&y, 0); daceAllocateDA(&t, 0);
    daceCreateVariable(&x, 1, 1.0);
    daceCreateVariable(&y, 2, 1.0);

    /* f = 1.7 + 0.9 x - 0.2 y + 0.15 xy */
    daceCreateConstant(&f, 1.7);
    daceMultiplyDouble(&x, 0.9, &t); daceAdd(&f, &t, &f);
    daceMultiplyDouble(&y, -0.2, &t); daceAdd(&f, &t, &f);
    daceMultiply(&x, &y, &t); daceMultiplyDouble(&t, 0.15, &t); daceAdd(&f, &t, &f);

    daceBesselJFunction(&f, 1, &g); dump(&g); printf("====\n");
    daceBesselYFunction(&f, 1, &g); dump(&g); printf("====\n");
    daceBesselIFunction(&f, 1, 0, &g); dump(&g); printf("====\n");
    daceBesselKFunction(&f, 1, 0, &g); dump(&g); printf("====\n");
    daceBesselIFunction(&f, 2, 1, &g); dump(&g); printf("====\n");
    daceBesselKFunction(&f, 2, 1, &g); dump(&g); printf("====\n");
    daceLogGammaFunction(&f, &g); dump(&g); printf("====\n");
    daceGammaFunction(&f, &g); dump(&g); printf("====\n");
    dacePsiFunction(&f, 0, &g); dump(&g); printf("====\n");
    dacePsiFunction(&f, 2, &g); dump(&g);
    return 0;
}
