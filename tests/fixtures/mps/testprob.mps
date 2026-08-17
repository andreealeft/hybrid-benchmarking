* The worked example from the classic MPS documentation.
*   minimise  XONE + 2 YTWO + 3 ZTHREE
*   LIM1:  XONE + YTWO            <=  4
*   LIM2:  XONE          + ZTHREE >=  1
*   MYEQN:      - YTWO  + ZTHREE   =  7
*   0 <= XONE <= 4,  -1 <= YTWO,  0 <= ZTHREE
NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST             1.0   LIM1             1.0
    XONE      LIM2             1.0
    YTWO      COST             2.0   LIM1             1.0
    YTWO      MYEQN           -1.0
    ZTHREE    COST             3.0   LIM2             1.0
    ZTHREE    MYEQN            1.0
RHS
    RHS       LIM1             4.0   LIM2             1.0
    RHS       MYEQN            7.0
BOUNDS
 UP BND       XONE             4.0
 LO BND       YTWO            -1.0
ENDATA
