* Two N rows.  COST is the objective because it comes first; TALLY is a free
* row -- an expression the modeller wanted reported, constraining nothing -- and
* by convention it is dropped along with everything written into it.  A reader
* that added TALLY's coefficients to the objective would return a different
* model that still solves.
NAME          TWONROWS
ROWS
 N  COST
 N  TALLY
 L  CAP
COLUMNS
    XA        COST             1.0   TALLY          100.0
    XA        CAP              1.0
    XB        TALLY          200.0   CAP              1.0
RHS
    RHS       CAP              9.0   TALLY           50.0
ENDATA
