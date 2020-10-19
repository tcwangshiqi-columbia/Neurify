/* Objective function */
max: +3 money +3 C2 +3 C3;

/* Constraints */
speed: +4 money +C2 +2 C3 >= 3;
+2 money +4 C2 +2 C3 = 8;

/* Variable bounds */
money >= 2;
C2 <= 5.3;
