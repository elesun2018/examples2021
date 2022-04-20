#include <stdio.h>
#include "test_so.h"
 
void main()
{
    printf("%s():%d\n",__func__, __LINE__);
    test_a();
    test_b();
}