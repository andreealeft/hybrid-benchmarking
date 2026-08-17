* Negative UP bounds.  XFREE gets one with no lower bound stated, so its lower
* bound is released to minus infinity; XZERO has its zero lower bound written
* down first, so the zero stands and the column is infeasible as written, which
* is the modeller's business and not the reader's; XPLAIN is the ordinary case.
NAME          NEGUP
ROWS
 N  COST
 G  KEEP
COLUMNS
    XFREE     COST             1.0   KEEP             1.0
    XZERO     COST             1.0   KEEP             1.0
    XPLAIN    COST             1.0   KEEP             1.0
RHS
    RHS       KEEP            -6.0
BOUNDS
 UP BND       XFREE           -3.0
 LO BND       XZERO            0.0
 UP BND       XZERO           -2.0
 UP BND       XPLAIN           5.0
ENDATA
