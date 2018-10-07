#include <stdlib.h>
#include <stdio.h>

int EndOfPgr(int i)
{
  exit(i);
}

void press_ret(void)
{
  printf("[return]");
  getchar();
}

int main(void)
{
#if defined EXPLICIT
  extern int demoExplicit();

  return(demoExplicit());
#else
  extern int demoImplicit();

  return(demoImplicit());
#endif
}
