#include <stdio.h>
#include "fun.h"

void function(unsigned char cnt)
{
	unsigned char i,sum=0;
	for(i = 0;i < cnt;i ++)
		{
	          sum = sum + i;
		  printf("sum of number from 0 to %d is %d\r\n",i,sum); 
		}
}

