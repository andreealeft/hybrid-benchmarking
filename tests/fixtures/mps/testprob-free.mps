* The same model, written free: fields separated by whitespace alone, with
* the card columns ignored and the row/value pairs grouped differently.
NAME TESTPROB
ROWS
 N COST
 L LIM1
 G LIM2
 E MYEQN
COLUMNS
 XONE COST 1.0
 XONE LIM1 1.0 LIM2 1.0
 YTWO COST 2.0 LIM1 1.0
 YTWO MYEQN -1.0
 ZTHREE COST 3.0
 ZTHREE LIM2 1.0 MYEQN 1.0
RHS
 RHS LIM1 4.0
 RHS LIM2 1.0 MYEQN 7.0
BOUNDS
 UP BND XONE 4.0
 LO BND YTWO -1.0
ENDATA
