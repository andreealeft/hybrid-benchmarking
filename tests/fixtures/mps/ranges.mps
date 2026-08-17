* A model with a RANGES section, one range on each kind of row that can carry
* one, and one row left without.  The signed range on the equality row is the
* case whose interpretation depends on the sign, which is why the reader keeps
* the raw number.
NAME          RANGED
ROWS
 N  COST
 L  RLESS
 G  RMORE
 E  REQUAL
 L  RPLAIN
COLUMNS
    XA        COST             1.0   RLESS            1.0
    XA        RMORE            1.0   REQUAL           1.0
    XB        COST             1.0   RPLAIN           1.0
    XB        RLESS            2.0
RHS
    RHS       RLESS           10.0   RMORE            2.0
    RHS       REQUAL           5.0   RPLAIN           8.0
RANGES
    RNG       RLESS            4.0   RMORE            6.0
    RNG       REQUAL          -3.0
ENDATA
