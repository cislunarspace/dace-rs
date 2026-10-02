/* Dump several elementary function results at (10,2) via daceWrite. */
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
    daceInitialize(10, 2);
    DACEDA f, g, t;
    daceAllocateDA(&f, 0); daceAllocateDA(&g, 0); daceAllocateDA(&t, 0);

    /* f = 0.7 + 1.3 x - 0.4 y + 0.25 xy */
    DACEDA x, y;
    daceAllocateDA(&x, 0); daceAllocateDA(&y, 0);
    daceCreateVariable(&x, 1, 1.0);
    daceCreateVariable(&y, 2, 1.0);
    daceCreateConstant(&f, 0.7);
    daceMultiplyDouble(&x, 1.3, &t); daceAdd(&f, &t, &f);
    daceMultiplyDouble(&y, -0.4, &t); daceAdd(&f, &t, &f);
    daceMultiply(&x, &y, &t); daceMultiplyDouble(&t, 0.25, &t); daceAdd(&f, &t, &f);

    daceSine(&f, &g); dump(&g); printf("====\n");
    daceExponential(&f, &g); dump(&g); printf("====\n");
    daceLogarithm(&f, &g); dump(&g); printf("====\n");
    daceTangent(&f, &g); dump(&g); printf("====\n");
    daceSquareRoot(&f, &g); dump(&g); printf("====\n");
    daceArcTangent(&f, &g); dump(&g); printf("====\n");
    daceErrorFunction(&f, &g); dump(&g); printf("====\n");
    dacePower(&f, 3, &g); dump(&g); printf("====\n");
    daceRoot(&f, 3, &g); dump(&g); printf("====\n");
    daceHyperbolicTangent(&f, &g); dump(&g); printf("====\n");
    daceArcSine(&f, &g); dump(&g); printf("====\n");
    daceArcTangent2(&f, &x, &g); dump(&g);
    return 0;
}
