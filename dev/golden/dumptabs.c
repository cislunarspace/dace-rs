/* Dump DACE addressing tables for parity checking against dace-rs. */
#include <stdio.h>
#include "dace/dacebase.h"
#include "dace/daceaux.h"

int main(void)
{
    unsigned int no, nv;
    while (scanf("%u %u", &no, &nv) == 2) {
        daceInitialize(no, nv);
        printf("CONTEXT %u %u %u %u %u %u\n", DACECom.nomax, DACECom.nvmax,
               DACECom.nv1, DACECom.nv2, DACECom.nmmax, (unsigned)(DACECom.epsmac * 1e16));
        for (unsigned int i = 0; i < DACECom.nmmax; i++)
            printf("IE %u %u %u %u\n", i, DACECom.ie1[i], DACECom.ie2[i], DACECom.ieo[i]);
        unsigned int lia = 1;
        for (unsigned int k = 0; k < DACECom.nv1; k++) lia *= (DACECom.nomax + 1);
        for (unsigned int i = 0; i <= lia; i++)
            printf("IA %u %u %u\n", i, DACECom.ia1[i], DACECom.ia2[i]);
    }
    return 0;
}
