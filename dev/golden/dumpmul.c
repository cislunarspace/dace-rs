/* Multiply two fixed DAs at (6,3) and dump the result via daceWrite. */
#include <stdio.h>
#include "dace/dacebase.h"

static void dump(const DACEDA *d)
{
    char buf[140 * 4096];
    unsigned int n = 0;
    daceWrite(d, buf, &n);
    for (unsigned int i = 0; i < n; i++)
        printf("%s\n", &buf[i * 140]);
}

int main(void)
{
    daceInitialize(6, 3);
    DACEDA a, b, c, t;
    daceAllocateDA(&a, 0); daceAllocateDA(&b, 0); daceAllocateDA(&c, 0);
    daceAllocateDA(&t, 0);

    unsigned int jjc[3] = {0, 0, 0};
    unsigned int jj1[3] = {1, 0, 0}, jj2[3] = {0, 1, 0}, jj3[3] = {0, 0, 1};
    unsigned int jj4[3] = {1, 1, 0}, jj5[3] = {0, 0, 2}, jj6[3] = {2, 0, 1};

    /* a = 1 + 2y + 0.3z - 1.7xy */
    daceCreateMonomial(&a, jjc, 1.0);
    daceCreateMonomial(&t, jj2, 2.0); daceAdd(&a, &t, &a);
    daceCreateMonomial(&t, jj3, 0.3); daceAdd(&a, &t, &a);
    daceCreateMonomial(&t, jj4, -1.7); daceAdd(&a, &t, &a);

    /* b = 0.5 + 1.3z^2 - 0.05 x^2 z */
    daceCreateMonomial(&b, jjc, 0.5);
    daceCreateMonomial(&t, jj5, 1.3); daceAdd(&b, &t, &b);
    daceCreateMonomial(&t, jj6, -0.05); daceAdd(&b, &t, &b);

    daceMultiply(&a, &b, &c);
    dump(&c);
    printf("====\n");
    daceMultiplicativeInverse(&b, &c);
    dump(&c);
    printf("====\n");
    daceDivide(&a, &b, &c);
    dump(&c);
    return 0;
}
