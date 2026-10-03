/* Golden parity generator: exercises the C DACE core over a fixed case
 * table and prints results in a line protocol consumed by tests/parity.rs.
 *
 * Protocol:
 *   INIT <no> <nv>
 *   CASE <name> [args...]
 *   IN <k> <jj...> <coeff>      (input DA k, built from daceCreateMonomial rows)
 *   SCA <value>                 (scalar result)
 *   BLOB <hex bytes>            (blob result)
 *   RES <jj...> <coeff>         (output DA row, in daceWrite order)
 * A case ends at the next CASE/INIT/EOF.
 * Build via dev/golden/gen.sh.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dace/dacebase.h"

static DACEDA A, B, C, T;
static char buf[140 * 65536];

static void init_das(void)
{
    daceAllocateDA(&A, 0); daceAllocateDA(&B, 0); daceAllocateDA(&C, 0);
    daceAllocateDA(&T, 0);
}

/* Build input DA k (0 or 1) from a monomial list. */
static void emit_in(int k, unsigned int *jj, int nv, double c)
{
    printf("IN %d", k);
    for (int i = 0; i < nv; i++) printf(" %u", jj[i]);
    printf(" %.17g\n", c);
}

static void emit_da(const DACEDA *d)
{
    unsigned int n = 0;
    daceWrite(d, buf, &n);
    for (unsigned int i = 1; i < n - 1; i++) {
        const char *line = &buf[i * 140];
        /* "     I  <24 coeff> <ord>  e e ..." -> reorder to "jj... coeff" */
        char coeff[25]; char ord[5];
        strncpy(coeff, line + 8, 24); coeff[24] = 0;
        strncpy(ord, line + 32, 4); ord[4] = 0;
        int o; sscanf(ord, "%d", &o);
        printf("RES");
        for (int j = 0; j < (int)daceGetMaxVariables(); j++) {
            char e[3]; strncpy(e, line + 38 + 3 * j, 2); e[2] = 0;
            int v; sscanf(e, "%d", &v);
            printf(" %d", v);
        }
        printf(" %s\n", coeff);
    }
}

typedef void (*unary_fn)(const DACEDA *, DACEDA *);

static void run_unary(const char *name, unary_fn fn)
{
    printf("CASE %s\n", name);
    fn(&A, &C);
    emit_da(&C);
}

int main(void)
{
    /* two contexts: (6,3) and (20,6) */
    double inputs1[][6] = {
        {1.2, 0.8, -0.5, 0.3, 0.0, -0.15},  /* f1 = 1.2 + 0.8x - 0.5y + 0.3xz - 0.15y^3 */
    };
    (void)inputs1;

    daceInitialize(6, 3);
    init_das();
    {
        DACEDA x, y, z, t;
        daceAllocateDA(&x, 0); daceAllocateDA(&y, 0); daceAllocateDA(&z, 0);
        daceAllocateDA(&t, 0);
        daceCreateVariable(&x, 1, 1.0);
        daceCreateVariable(&y, 2, 1.0);
        daceCreateVariable(&z, 3, 1.0);

        /* A = 1.2 + 0.8x - 0.5y + 0.3xyz */
        daceCreateConstant(&A, 1.2);
        daceMultiplyDouble(&x, 0.8, &t); daceAdd(&A, &t, &A);
        daceMultiplyDouble(&y, -0.5, &t); daceAdd(&A, &t, &A);
        daceMultiply(&x, &y, &t); daceMultiply(&z, &t, &t); daceMultiplyDouble(&t, 0.3, &t);
        daceAdd(&A, &t, &A);

        /* B = 0.9 + 1.1y - 0.7xz */
        daceCreateConstant(&B, 0.9);
        daceMultiplyDouble(&y, 1.1, &t); daceAdd(&B, &t, &B);
        daceMultiply(&x, &z, &t); daceMultiplyDouble(&t, -0.7, &t); daceAdd(&B, &t, &B);

        printf("INIT 6 3\n");

        printf("CASE add\n"); daceAdd(&A, &B, &C); emit_da(&C);
        printf("CASE sub\n"); daceSubtract(&A, &B, &C); emit_da(&C);
        printf("CASE mul\n"); daceMultiply(&A, &B, &C); emit_da(&C);
        printf("CASE div\n"); daceDivide(&A, &B, &C); emit_da(&C);
        printf("CASE fma_2_3_p5\n"); daceWeightedSum(&A, 2.0, &B, 3.5, &C); emit_da(&C);
        printf("CASE mul_double\n"); daceMultiplyDouble(&A, -1.75, &C); emit_da(&C);
        printf("CASE div_double\n"); daceDivideDouble(&A, 2.5, &C); emit_da(&C);
        printf("CASE add_double\n"); daceAddDouble(&A, -0.45, &C); emit_da(&C);

        run_unary("exp", daceExponential);
        run_unary("log", daceLogarithm);
        run_unary("log10", daceLogarithm10);
        run_unary("log2", daceLogarithm2);
        printf("CASE log_base_2p5\n"); daceLogarithmBase(&A, 2.5, &C); emit_da(&C);
        run_unary("sin", daceSine);
        run_unary("cos", daceCosine);
        run_unary("tan", daceTangent);
        run_unary("asin", daceArcSine);
        run_unary("acos", daceArcCosine);
        run_unary("atan", daceArcTangent);
        printf("CASE atan2\n"); daceArcTangent2(&A, &B, &C); emit_da(&C);
        run_unary("sinh", daceHyperbolicSine);
        run_unary("cosh", daceHyperbolicCosine);
        run_unary("tanh", daceHyperbolicTangent);
        run_unary("asinh", daceHyperbolicArcSine);
        run_unary("acosh", daceHyperbolicArcCosine);
        run_unary("atanh", daceHyperbolicArcTangent);
        run_unary("erf", daceErrorFunction);
        run_unary("erfc", daceComplementaryErrorFunction);
        printf("CASE powi_3\n"); dacePower(&A, 3, &C); emit_da(&C);
        printf("CASE powi_m2\n"); dacePower(&A, -2, &C); emit_da(&C);
        printf("CASE powf_1p7\n"); dacePowerDouble(&A, 1.7, &C); emit_da(&C);
        printf("CASE root_3\n"); daceRoot(&A, 3, &C); emit_da(&C);
        printf("CASE root_m2\n"); daceRoot(&A, -2, &C); emit_da(&C);
        run_unary("sqrt", daceSquareRoot);
        run_unary("isrt", daceInverseSquareRoot);
        run_unary("cbrt", daceCubicRoot);
        run_unary("icrt", daceInverseCubicRoot);
        printf("CASE hypot\n"); daceHypotenuse(&A, &B, &C); emit_da(&C);
        run_unary("minv", daceMultiplicativeInverse);
        run_unary("sqr", daceSquare);
        printf("CASE trunc_const\n"); daceTruncate(&A, &C); emit_da(&C);
        printf("CASE round_const\n"); daceRound(&A, &C); emit_da(&C);
        printf("CASE modulo_2\n"); daceModulo(&A, 2.0, &C); emit_da(&C);

        printf("CASE deriv_2\n"); daceDifferentiate(2, &A, &C); emit_da(&C);
        printf("CASE integ_3\n"); daceIntegrate(3, &A, &C); emit_da(&C);
        printf("CASE trim_1_2\n"); daceTrim(&A, 1, 2, &C); emit_da(&C);
        printf("CASE divide_variable_1_1\n"); daceDivideByVariable(&A, 1, 1, &C); emit_da(&C);
        /* A has no x in every term (constant/y/z terms) => pick x*yz term only via B-style DA */
        printf("CASE multiply_monomials\n"); daceMultiplyMonomials(&A, &B, &C); emit_da(&C);

        /* norms */
        printf("CASE norm_0\n"); printf("SCA %.17g\n", daceNorm(&A, 0));
        printf("CASE norm_1\n"); printf("SCA %.17g\n", daceNorm(&A, 1));
        printf("CASE norm_2\n"); printf("SCA %.17g\n", daceNorm(&A, 2));
        {
            double on[7];
            printf("CASE order_norm_0_1\n");
            daceOrderedNorm(&A, 0, 1, on);
            for (int i = 0; i <= 6; i++) printf("SCA %.17g\n", on[i]);
            printf("CASE order_norm_2_0\n");
            daceOrderedNorm(&A, 2, 0, on);
            for (int i = 0; i <= 6; i++) printf("SCA %.17g\n", on[i]);
            double c[9], err[7];
            printf("CASE estim_norm\n");
            daceEstimate(&A, 0, 1, c, err, 8);
            for (int i = 0; i <= 8; i++) printf("SCA %.17g\n", c[i]);
            for (int i = 0; i <= 6; i++) printf("SCA %.17g\n", err[i]);
        }
        {
            double lo, hi;
            daceGetBounds(&A, &lo, &hi);
            printf("CASE bound\n"); printf("SCA %.17g\nSCA %.17g\n", lo, hi);
        }

        /* evaluation */
        {
            const DACEDA *das[2] = { &A, &B };
            double ac[2 * 4000];
            unsigned int nterm, nvar, nord;
            daceEvalTree(das, 2, ac, &nterm, &nvar, &nord);
            printf("CASE eval_tree_meta\n"); printf("SCA %u\nSCA %u\nSCA %u\n", nterm, nvar, nord);
            /* evaluate the tree at a fixed point (compiledDA::eval) */
            {
                double args[3] = { 1.3, -0.6, 0.45 };
                double *p = ac + 2;
                double xm[8]; xm[0] = 1.0;
                double res[2] = { p[0], p[1] }; p += 2;
                for (unsigned int i = 1; i < nterm; i++) {
                    unsigned int jl = (unsigned int)(*p); p++;
                    unsigned int jv = (unsigned int)(*p) - 1; p++;
                    if (jv < 3) xm[jl] = xm[jl-1]*args[jv]; else xm[jl] = 0;
                    for (int j = 0; j < 2; j++, p++) res[j] += xm[jl]*(*p);
                }
                printf("CASE eval_point\n");
                printf("SCA %.17g\nSCA %.17g\n", res[0], res[1]);
            }
        }
        printf("CASE plug_2_p7\n"); daceEvalVariable(&A, 2, 0.7, &C); emit_da(&C);
        printf("CASE scale_variable_1_p5\n"); daceScaleVariable(&A, 1, 1.5, &C); emit_da(&C);
        printf("CASE translate_1_1p5_m0p4\n"); daceTranslateVariable(&A, 1, 1.5, -0.4, &C); emit_da(&C);
        printf("CASE eval_monomials\n"); printf("SCA %.17g\n", daceEvalMonomials(&A, &B));

        /* special functions */
        printf("CASE bessel_j_1\n"); daceBesselJFunction(&A, 1, &C); emit_da(&C);
        printf("CASE bessel_y_1\n"); daceBesselYFunction(&A, 1, &C); emit_da(&C);
        printf("CASE bessel_i_1\n"); daceBesselIFunction(&A, 1, 0, &C); emit_da(&C);
        printf("CASE bessel_k_1\n"); daceBesselKFunction(&A, 1, 0, &C); emit_da(&C);
        printf("CASE bessel_i_2_scaled\n"); daceBesselIFunction(&A, 2, 1, &C); emit_da(&C);
        printf("CASE bessel_k_2_scaled\n"); daceBesselKFunction(&A, 2, 1, &C); emit_da(&C);
        run_unary("log_gamma", daceLogGammaFunction);
        run_unary("gamma", daceGammaFunction);
        printf("CASE psi_0\n"); dacePsiFunction(&A, 0, &C); emit_da(&C);
        printf("CASE psi_2\n"); dacePsiFunction(&A, 2, &C); emit_da(&C);

        /* blob */
        {
            unsigned int size = 0;
            daceExportBlob(&A, NULL, &size);
            unsigned char *blob = (unsigned char *)malloc(size);
            daceExportBlob(&A, blob, &size);
            printf("CASE blob\nBLOB ");
            for (unsigned int i = 0; i < size; i++) printf("%02x", blob[i]);
            printf("\n");
            free(blob);
        }
    }

    daceInitialize(20, 6);
    init_das();
    {
        DACEDA v[6], t;
        for (int i = 0; i < 6; i++) daceAllocateDA(&v[i], 0);
        daceAllocateDA(&t, 0);
        for (int i = 0; i < 6; i++) daceCreateVariable(&v[i], i + 1, 1.0);

        /* A = 1.5 - 0.6x2 + 0.35x1x3 - 0.2x4^2 + 0.1x5x6 */
        daceCreateConstant(&A, 1.5);
        daceMultiplyDouble(&v[1], -0.6, &t); daceAdd(&A, &t, &A);
        daceMultiply(&v[0], &v[2], &t); daceMultiplyDouble(&t, 0.35, &t); daceAdd(&A, &t, &A);
        daceMultiply(&v[3], &v[3], &t); daceMultiplyDouble(&t, -0.2, &t); daceAdd(&A, &t, &A);
        daceMultiply(&v[4], &v[5], &t); daceMultiplyDouble(&t, 0.1, &t); daceAdd(&A, &t, &A);

        /* B = 2.1 + 0.4x1 - 0.25x2x5 + 0.15x3^3 */
        daceCreateConstant(&B, 2.1);
        daceMultiplyDouble(&v[0], 0.4, &t); daceAdd(&B, &t, &B);
        daceMultiply(&v[1], &v[4], &t); daceMultiplyDouble(&t, -0.25, &t); daceAdd(&B, &t, &B);
        daceMultiply(&v[2], &v[2], &t); daceMultiply(&v[2], &t, &t); daceMultiplyDouble(&t, 0.15, &t);
        daceAdd(&B, &t, &B);

        printf("INIT 20 6\n");
        printf("CASE add\n"); daceAdd(&A, &B, &C); emit_da(&C);
        printf("CASE mul\n"); daceMultiply(&A, &B, &C); emit_da(&C);
        printf("CASE div\n"); daceDivide(&A, &B, &C); emit_da(&C);
        run_unary("exp", daceExponential);
        run_unary("sin", daceSine);
        run_unary("log", daceLogarithm);
        run_unary("tanh", daceHyperbolicTangent);
        run_unary("erf", daceErrorFunction);
        printf("CASE powf_0p5\n"); dacePowerDouble(&A, 0.5, &C); emit_da(&C);
        printf("CASE root_5\n"); daceRoot(&A, 5, &C); emit_da(&C);
        run_unary("gamma", daceGammaFunction);
        printf("CASE bessel_j_3\n"); daceBesselJFunction(&A, 3, &C); emit_da(&C);
        printf("CASE bessel_k_1_scaled\n"); daceBesselKFunction(&A, 1, 1, &C); emit_da(&C);
        printf("CASE atan2\n"); daceArcTangent2(&A, &B, &C); emit_da(&C);
        printf("CASE hypot\n"); daceHypotenuse(&A, &B, &C); emit_da(&C);
        printf("CASE norm_1\n"); printf("SCA %.17g\n", daceNorm(&A, 1));
        {
            double lo, hi;
            daceGetBounds(&A, &lo, &hi);
            printf("CASE bound\n"); printf("SCA %.17g\nSCA %.17g\n", lo, hi);
        }
        printf("CASE eval_monomials\n"); printf("SCA %.17g\n", daceEvalMonomials(&A, &B));
    }

    /* third context: constant-term choices for domain edges */
    daceInitialize(6, 2);
    init_das();
    {
        DACEDA x, y, t;
        daceAllocateDA(&x, 0); daceAllocateDA(&y, 0); daceAllocateDA(&t, 0);
        daceCreateVariable(&x, 1, 1.0);
        daceCreateVariable(&y, 2, 1.0);

        printf("INIT 6 2\n");
        /* log at 0.5 and 2.0 */
        daceCreateConstant(&A, 0.5); daceMultiplyDouble(&x, 0.3, &t); daceAdd(&A, &t, &A);
        daceMultiply(&y, &y, &t); daceMultiplyDouble(&t, -0.1, &t); daceAdd(&A, &t, &A);
        run_unary("log_c0p5", daceLogarithm);
        daceCreateConstant(&A, 2.0); daceMultiplyDouble(&x, -0.4, &t); daceAdd(&A, &t, &A);
        daceMultiply(&x, &y, &t); daceMultiplyDouble(&t, 0.2, &t); daceAdd(&A, &t, &A);
        run_unary("log_c2", daceLogarithm);
        /* sqrt at 4.0 */
        daceCreateConstant(&A, 4.0); daceMultiplyDouble(&y, 0.25, &t); daceAdd(&A, &t, &A);
        run_unary("sqrt_c4", daceSquareRoot);
        /* bessel at 1.7 */
        daceCreateConstant(&A, 1.7); daceMultiplyDouble(&x, 0.2, &t); daceAdd(&A, &t, &A);
        daceMultiplyDouble(&y, -0.15, &t); daceAdd(&A, &t, &A);
        printf("CASE bessel_j_0_c1p7\n"); daceBesselJFunction(&A, 0, &C); emit_da(&C);
        /* gamma at 0.5 */
        daceCreateConstant(&A, 0.5); daceMultiplyDouble(&x, 0.1, &t); daceAdd(&A, &t, &A);
        run_unary("gamma_c0p5", daceGammaFunction);
        /* psi at 2 */
        daceCreateConstant(&A, 2.0); daceMultiplyDouble(&x, -0.05, &t); daceAdd(&A, &t, &A);
        printf("CASE psi_2_c2\n"); dacePsiFunction(&A, 2, &C); emit_da(&C);
        /* sin at pi/2-ish constant */
        daceCreateConstant(&A, 1.5707963267948966); daceMultiplyDouble(&x, 0.1, &t); daceAdd(&A, &t, &A);
        run_unary("sin_cpiover2", daceSine);
        run_unary("tan_cpiover2", daceTangent);
    }

    return 0;
}
