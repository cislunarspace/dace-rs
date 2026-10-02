/* Dump norms, bounds, eval-tree results for fixed inputs. */
#include <stdio.h>
#include "dace/dacebase.h"

int main(void)
{
    daceInitialize(6, 3);
    DACEDA x, y, z, f, t, g;
    daceAllocateDA(&x, 0); daceAllocateDA(&y, 0); daceAllocateDA(&z, 0);
    daceAllocateDA(&f, 0); daceAllocateDA(&t, 0); daceAllocateDA(&g, 0);
    daceCreateVariable(&x, 1, 1.0);
    daceCreateVariable(&y, 2, 1.0);
    daceCreateVariable(&z, 3, 1.0);

    /* f = 1.2 + 0.8x - 0.5y + 0.3xz - 0.2y^2 */
    daceCreateConstant(&f, 1.2);
    daceMultiplyDouble(&x, 0.8, &t); daceAdd(&f, &t, &f);
    daceMultiplyDouble(&y, -0.5, &t); daceAdd(&f, &t, &f);
    daceMultiply(&x, &z, &t); daceMultiplyDouble(&t, 0.3, &t); daceAdd(&f, &t, &f);
    daceMultiply(&y, &y, &t); daceMultiplyDouble(&t, -0.2, &t); daceAdd(&f, &t, &f);

    printf("NORM %.17g\n", daceNorm(&f, 0));
    printf("NORM %.17g\n", daceNorm(&f, 1));
    printf("NORM %.17g\n", daceNorm(&f, 2));
    double on[7];
    daceOrderedNorm(&f, 0, 1, on);
    for (int i = 0; i <= 6; i++) printf("ON %d %.17g\n", i, on[i]);
    daceOrderedNorm(&f, 2, 0, on);
    for (int i = 0; i <= 6; i++) printf("OV %d %.17g\n", i, on[i]);
    double c[9], err[7];
    daceEstimate(&f, 0, 1, c, err, 8);
    for (int i = 0; i <= 8; i++) printf("EST %d %.17g\n", i, c[i]);
    for (int i = 0; i <= 6; i++) printf("ERR %d %.17g\n", i, err[i]);
    double lo, hi;
    daceGetBounds(&f, &lo, &hi);
    printf("BND %.17g %.17g\n", lo, hi);

    /* eval tree */
    const DACEDA *das[2] = { &f, &g };
    daceMultiplyDouble(&y, 2.0, &g); daceAdd(&z, &g, &g);
    double ac[2 * 200 * (2 + 2)];
    unsigned int nterm, nvar, nord;
    daceEvalTree(das, 2, ac, &nterm, &nvar, &nord);
    printf("TREE %u %u %u\n", nterm, nvar, nord);
    double args[3] = { 1.3, -0.6, 0.45 };
    /* evaluate tree manually (compiledDA::eval algorithm) */
    {
        double *p = ac + 2;
        double xm[8]; xm[0] = 1.0;
        double res[2] = { p[0], p[1] }; p += 2;
        for (unsigned int i = 1; i < nterm; i++) {
            unsigned int jl = (unsigned int)(*p); p++;
            unsigned int jv = (unsigned int)(*p) - 1; p++;
            if (jv < 3) xm[jl] = xm[jl-1]*args[jv]; else xm[jl] = 0;
            for (int j = 0; j < 2; j++, p++) res[j] += xm[jl]*(*p);
        }
        printf("EVAL %.17g %.17g\n", res[0], res[1]);
    }

    /* plug / replace / scale / translate / evalMonomials */
    daceEvalVariable(&f, 2, 0.7, &t);
    {
        char buf[140 * 8192]; unsigned int n = 0;
        daceWrite(&t, buf, &n);
        for (unsigned int i = 0; i < n; i++) printf("%s\n", &buf[i * 140]);
    }
    printf("====\n");
    daceReplaceVariable(&f, 3, 1, 2.5, &t);
    {
        char buf[140 * 8192]; unsigned int n = 0;
        daceWrite(&t, buf, &n);
        for (unsigned int i = 0; i < n; i++) printf("%s\n", &buf[i * 140]);
    }
    printf("====\n");
    daceTranslateVariable(&f, 1, 1.5, -0.4, &t);
    {
        char buf[140 * 8192]; unsigned int n = 0;
        daceWrite(&t, buf, &n);
        for (unsigned int i = 0; i < n; i++) printf("%s\n", &buf[i * 140]);
    }
    printf("====\n");
    printf("EVM %.17g\n", daceEvalMonomials(&f, &f));
    return 0;
}
